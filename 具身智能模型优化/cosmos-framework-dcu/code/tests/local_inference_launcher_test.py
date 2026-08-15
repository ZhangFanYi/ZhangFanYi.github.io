# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Contract tests for the local Generator inference launchers.

These tests replace ``torchrun`` with a tiny executable that records argv, so
they validate wrapper behavior without loading a model or requiring GPUs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class LauncherSpec:
    name: str
    nano_wrapper: str
    super_wrapper: str
    visible_devices_var: str
    i2v_input: str


LAUNCHERS = (
    LauncherSpec(
        name="hcu",
        nano_wrapper="launch_inference_generator_vision_nano_hcu.sh",
        super_wrapper="launch_inference_generator_vision_super_hcu.sh",
        visible_devices_var="HIP_VISIBLE_DEVICES",
        i2v_input="inputs/omni/i2v_local_hcu.json",
    ),
    LauncherSpec(
        name="h20",
        nano_wrapper="launch_inference_generator_vision_nano_nvidia_local.sh",
        super_wrapper="launch_inference_generator_vision_super_nvidia_local.sh",
        visible_devices_var="CUDA_VISIBLE_DEVICES",
        i2v_input="inputs/omni/i2v_local_nvidia.json",
    ),
)


def _make_fake_runtime(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_args = tmp_path / "torchrun_args.txt"
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURED_TORCHRUN_ARGS"\n')
    fake_torchrun.chmod(0o755)

    input_file = tmp_path / "t2v.json"
    input_file.write_text("{}")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}")
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir()
    vae_path = tmp_path / "wan_vae.pth"
    vae_path.touch()
    local_asset_launcher = tmp_path / "local_assets.py"
    local_asset_launcher.write_text("# fake launcher\n")

    return fake_bin, captured_args, {
        "INPUT_FILE": str(input_file),
        "CHECKPOINT_PATH": str(checkpoint_dir),
        "PROCESSOR_DIR": str(processor_dir),
        "WAN_VAE_PATH": str(vae_path),
        "LOCAL_ASSET_LAUNCHER": str(local_asset_launcher),
    }


def _launcher_env(
    spec: LauncherSpec,
    fake_bin: Path,
    captured_args: Path,
    assets: dict[str, str],
    tmp_path: Path,
    **overrides: str,
) -> dict[str, str]:
    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "CAPTURED_TORCHRUN_ARGS": str(captured_args),
        spec.visible_devices_var: "0,1,2,3,4,5,6,7",
        "MASTER_PORT": "30799",
        "INFERENCE_PROFILE": "official-t2v",
        "OUTPUT_DIR": str(tmp_path / "output"),
        "GUARDRAILS": "false",
        "BENCHMARK": "false",
        "HF_HUB_OFFLINE": "1",
        "COSMOS_SMOKE": "0",
        **assets,
    }
    env.update(overrides)
    return env


def _run_wrapper(wrapper: Path, repo_root: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(wrapper), *args],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize("spec", LAUNCHERS, ids=lambda spec: spec.name)
def test_multigpu_latency_leaves_dp_sharding_to_framework(tmp_path: Path, spec: LauncherSpec) -> None:
    """Neither platform wrapper may override the official FSDP auto-plan."""
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "examples" / spec.nano_wrapper
    fake_bin, captured_args, assets = _make_fake_runtime(tmp_path)
    env = _launcher_env(spec, fake_bin, captured_args, assets, tmp_path)

    result = _run_wrapper(wrapper, repo_root, env, "--warmup=0")

    assert result.returncode == 0, result.stderr
    args = captured_args.read_text().splitlines()
    assert "--parallelism-preset=latency" in args
    assert "--resolution=720" in args
    assert "--num-frames=189" in args
    assert "--no-guardrails" in args
    assert not any(arg.startswith("--dp-shard-size=") for arg in args)


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize("spec", LAUNCHERS, ids=lambda spec: spec.name)
def test_launcher_defaults_to_in_repo_asset_shim(tmp_path: Path, spec: LauncherSpec) -> None:
    """The committed launcher must not depend on a sibling workspace checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "examples" / spec.nano_wrapper
    fake_bin, captured_args, assets = _make_fake_runtime(tmp_path)
    assets.pop("LOCAL_ASSET_LAUNCHER")
    env = _launcher_env(spec, fake_bin, captured_args, assets, tmp_path)

    result = _run_wrapper(wrapper, repo_root, env, "--warmup=0")

    assert result.returncode == 0, result.stderr
    args = captured_args.read_text().splitlines()
    expected_suffix = f"/{repo_root.name}/tools/run_framework_inference_local_assets.py"
    assert any(arg.replace("\\", "/").endswith(expected_suffix) for arg in args)


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize("spec", LAUNCHERS, ids=lambda spec: spec.name)
def test_super_i2v_delegates_without_reintroducing_dp_override(tmp_path: Path, spec: LauncherSpec) -> None:
    """Super selects its platform I2V input and preserves the Nano contract."""
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "examples" / spec.super_wrapper
    fake_bin, captured_args, assets = _make_fake_runtime(tmp_path)
    assets.pop("INPUT_FILE")
    env = _launcher_env(
        spec,
        fake_bin,
        captured_args,
        assets,
        tmp_path,
        INFERENCE_PROFILE="official-i2v",
    )

    result = _run_wrapper(wrapper, repo_root, env, "--warmup=0")

    assert result.returncode == 0, result.stderr
    args = captured_args.read_text().splitlines()
    assert args[args.index("-i") + 1] == spec.i2v_input
    assert "--parallelism-preset=latency" in args
    assert not any(arg.startswith("--dp-shard-size=") for arg in args)


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize("spec", LAUNCHERS, ids=lambda spec: spec.name)
def test_guardrail_dependency_failure_happens_before_torchrun(tmp_path: Path, spec: LauncherSpec) -> None:
    """A missing Guardrails extra must fail before distributed model startup."""
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "examples" / spec.nano_wrapper
    fake_bin, captured_args, assets = _make_fake_runtime(tmp_path)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 1\n")
    fake_python.chmod(0o755)
    guardrail_dir = tmp_path / "guardrail"
    qwen_dir = tmp_path / "qwen"
    guardrail_dir.mkdir()
    qwen_dir.mkdir()
    env = _launcher_env(
        spec,
        fake_bin,
        captured_args,
        assets,
        tmp_path,
        GUARDRAILS="true",
        COSMOS_GUARDRAIL1_PATH=str(guardrail_dir),
        COSMOS_QWEN3GUARD_PATH=str(qwen_dir),
    )

    result = _run_wrapper(wrapper, repo_root, env)

    assert result.returncode == 2
    assert "retinaface-py" in result.stderr
    assert not captured_args.exists()
