#!/usr/bin/env python3
"""Run the official Cosmos Framework inference entrypoint with local Guardrail assets.

Without either path environment variable this is intentionally equivalent to
``python -m cosmos_framework.scripts.inference``.  When paths are supplied,
only model-resource resolution is overridden: the official Guardrail presets
and inference logic remain unchanged.

Environment variables:
    COSMOS_GUARDRAIL1_PATH: local ``nvidia/Cosmos-Guardrail1`` repository.
    COSMOS_QWEN3GUARD_PATH: local ``Qwen/Qwen3Guard-Gen-0.6B`` repository.
    WAN_VAE_PATH: local ``Wan2.2_VAE.pth`` file.
    PROCESSOR_DIR: local Cosmos3-Nano/Super processor directory.

``COSMOS_HCU_QWEN3GUARD_DIR`` remains accepted as a legacy alias for the
second variable used by ``tools/run_hcu_framework_inference.py``.
"""
from __future__ import annotations

import json
import os
import runpy
import shutil
import time
import contextlib
from pathlib import Path


def _local_directory(*names: str) -> Path | None:
    configured = [(name, os.environ[name]) for name in names if os.environ.get(name)]
    if not configured:
        return None

    values = {value for _, value in configured}
    if len(values) != 1:
        rendered = ", ".join(f"{name}={value}" for name, value in configured)
        raise ValueError(f"Conflicting local model paths: {rendered}")

    path = Path(configured[0][1]).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{configured[0][0]} is not a directory: {path}")
    return path


def patch_hcu_python_compat() -> None:
    """Apply the small Python 3.10/HCU compatibility surface before imports.

    The HCU dependency overlay supplies its own Transformers package while the
    container may supply a newer system ``huggingface_hub``.  The historical
    HCU Framework launcher patched this boundary before importing any module
    that transitively imports Transformers; keep that ordering here too.
    """
    if not hasattr(contextlib, "chdir"):
        @contextlib.contextmanager
        def chdir(path):
            old_cwd = os.getcwd()
            os.chdir(path)
            try:
                yield
            finally:
                os.chdir(old_cwd)

        contextlib.chdir = chdir  # type: ignore[attr-defined]

    try:
        import huggingface_hub
    except ImportError:
        return

    if not hasattr(huggingface_hub, "is_offline_mode"):
        def is_offline_mode() -> bool:
            return os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}

        huggingface_hub.is_offline_mode = is_offline_mode  # type: ignore[attr-defined]


def _prepare_local_sound_tokenizer(local_sound_tokenizer: Path, materialize) -> Path:
    """Build a complete legacy AVAE cache from the local HF files.

    ``torchrun`` starts this shim independently in every rank.  Only rank 0
    writes the shared cache; files are copied through a temporary name and
    atomically renamed so other ranks never load a partial safetensors file.
    """
    legacy_ckpt = local_sound_tokenizer / "avae_48k_noncausal_25hz_64ch.ckpt"
    legacy_json = local_sound_tokenizer / "avae_48k_noncausal_25hz_64ch.json"
    if legacy_ckpt.exists() and legacy_json.exists():
        return local_sound_tokenizer

    cache_dir = Path(__file__).resolve().parents[1] / "outputs" / "hcu_framework" / "sound_tokenizer_legacy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RANK", "0") == "0":
        for name in ("config.json", "diffusion_pytorch_model.safetensors", "model.safetensors"):
            src = local_sound_tokenizer / name
            if not src.exists():
                continue
            dst = cache_dir / name
            tmp = cache_dir / f".{name}.tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        materialize(str(cache_dir))
    else:
        deadline = time.monotonic() + 180.0
        while not (
            (cache_dir / "avae_48k_noncausal_25hz_64ch.ckpt").exists()
            and (cache_dir / "avae_48k_noncausal_25hz_64ch.json").exists()
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for AVAE cache: {cache_dir}")
            time.sleep(0.5)
    return cache_dir


def _checkpoint_has_visual_weights() -> bool:
    """Return whether the selected local checkpoint contains the VLM tower.

    The converted Super DCP used by the HCU training recipe intentionally
    stores the generation model without ``net.language_model.visual.*``.
    Nano DCP and the local HF Generator checkpoints do contain those keys.
    The inference config must match the state-dict shape before DCP loading;
    otherwise the loader asks for visual projection biases that are not in the
    converted Super checkpoint.
    """
    value = os.environ.get("CHECKPOINT_PATH") or os.environ.get("COSMOS_CHECKPOINT_PATH")
    if not value:
        return True

    checkpoint = Path(value).expanduser()
    metadata_candidates = (
        checkpoint / "model" / ".metadata",
        checkpoint / ".metadata",
    )
    for metadata in metadata_candidates:
        if metadata.is_file():
            try:
                return b"net.language_model.visual." in metadata.read_bytes()
            except OSError:
                return True

    index_candidates = (
        checkpoint / "model.safetensors.index.json",
        checkpoint / "model" / "model.safetensors.index.json",
    )
    for index in index_candidates:
        if not index.is_file():
            continue
        try:
            index_data = json.loads(index.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
            if not isinstance(weight_map, dict):
                return True

            # Standard HF VLM checkpoints expose language_model/model.visual
            # keys.  Cosmos3 Diffusers checkpoints keep the same tower under
            # ``vision_encoder/`` and use bare ViT keys (blocks, merger,
            # patch_embed, ...); the Framework Diffusers load planner maps
            # those keys to ``language_model.visual.*`` at load time.
            visual_key_prefixes = (
                "language_model.visual.",
                "model.visual.",
                "visual.",
                "blocks.",
                "deepstack_merger_list.",
                "merger.",
                "patch_embed.",
                "pos_embed.",
            )
            for key, relative_path in weight_map.items():
                normalized_path = str(relative_path).replace("\\", "/")
                if normalized_path.startswith("vision_encoder/"):
                    return True
                if str(key).startswith(visual_key_prefixes):
                    return True
            return False
        except (OSError, json.JSONDecodeError, AttributeError):
            return True

    # A non-indexed local HF directory or an unresolved registry path keeps
    # the historical visual default; the official resolver will report a
    # missing asset if the path is not usable.
    return True


def patch_local_generator_assets() -> None:
    """Route the Generator VAE and VLM processor to explicit local assets.

    The official inference CLI intentionally has no training-only Hydra
    override surface when ``COSMOS_TRAINING=false``.  Keep the CLI official
    and patch only the two local resource resolvers that otherwise invoke the
    checkpoint/Hugging Face download path.
    """
    vae_value = os.environ.get("WAN_VAE_PATH") or os.environ.get("COSMOS_HCU_WAN22_VAE")
    processor_value = os.environ.get("PROCESSOR_DIR") or os.environ.get("COSMOS_HCU_PROCESSOR_DIR")

    local_vae = Path(vae_value).expanduser().resolve() if vae_value else None
    local_processor = Path(processor_value).expanduser().resolve() if processor_value else None
    if local_vae is not None and not local_vae.is_file():
        raise FileNotFoundError(f"WAN_VAE_PATH is not a file: {local_vae}")

    import cosmos_framework.utils.checkpoint_db as checkpoint_db
    from cosmos_framework.inference.common.checkpoints import _materialize_avae_ckpt

    sound_value = (
        os.environ.get("SOUND_TOKENIZER_DIR")
        or os.environ.get("COSMOS_HCU_SOUND_TOKENIZER_DIR")
        or (str(local_processor / "sound_tokenizer") if local_processor is not None else "")
    )
    local_sound_tokenizer = Path(sound_value).expanduser().resolve() if sound_value else None
    materialized_sound_tokenizer: Path | None = None
    if local_sound_tokenizer is not None and local_sound_tokenizer.is_dir():
        legacy_ckpt = local_sound_tokenizer / "avae_48k_noncausal_25hz_64ch.ckpt"
        legacy_json = local_sound_tokenizer / "avae_48k_noncausal_25hz_64ch.json"
        if legacy_ckpt.exists() and legacy_json.exists():
            materialized_sound_tokenizer = local_sound_tokenizer
        else:
            materialized_sound_tokenizer = _prepare_local_sound_tokenizer(
                local_sound_tokenizer,
                _materialize_avae_ckpt,
            )

    original_download_checkpoint_v2 = checkpoint_db.download_checkpoint_v2

    def download_checkpoint_v2_local(uri: str, *, check_exists: bool = True) -> str:
        if local_vae is not None and uri.endswith("pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"):
            return str(local_vae)
        if materialized_sound_tokenizer is not None and uri.endswith("pretrained/tokenizers/audio/avae"):
            return str(materialized_sound_tokenizer)
        return original_download_checkpoint_v2(uri, check_exists=check_exists)

    checkpoint_db.download_checkpoint_v2 = download_checkpoint_v2_local
    if local_vae is not None:
        print(f">>> Wan VAE local path: {local_vae}", flush=True)
    if materialized_sound_tokenizer is not None:
        print(f">>> AVAE local path: {materialized_sound_tokenizer}", flush=True)

    if local_processor is None:
        return

    if not local_processor.is_dir():
        raise FileNotFoundError(f"PROCESSOR_DIR is not a directory: {local_processor}")

    import cosmos_framework.data.generator.processors as processors

    original_build_processor = processors.build_processor
    original_build_processor_lazy = processors.build_processor_lazy

    def build_processor_local(tokenizer_type: str, *args, **kwargs):
        if not Path(tokenizer_type).is_dir():
            tokenizer_type = str(local_processor)
        return original_build_processor(tokenizer_type, *args, **kwargs)

    def build_processor_lazy_local(*args, repository=None, revision=None, subdir="", **kwargs):
        local_path = local_processor / subdir if subdir else local_processor
        if local_path.is_dir() and (repository is not None or args):
            return original_build_processor(str(local_path), **kwargs)
        return original_build_processor_lazy(*args, repository=repository, revision=revision, subdir=subdir, **kwargs)

    processors.build_processor = build_processor_local
    processors.build_processor_lazy = build_processor_lazy_local

    # The Generator model config uses this resolver directly for the VLM
    # tokenizer.  Patching only build_processor is insufficient because the
    # resolver otherwise downloads Qwen3-VL tokenizer/config files before the
    # processor builder is reached.
    import cosmos_framework.configs.base.defaults.reasoner as reasoner_defaults

    original_create_qwen2_tokenizer = reasoner_defaults.create_qwen2_tokenizer_with_download

    def create_qwen2_tokenizer_local(
        pretrained_model_name: str,
        config_variant: str,
        *args,
        **kwargs,
    ):
        from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor

        if pretrained_model_name.startswith("Qwen/Qwen3-VL-"):
            return Qwen3VLProcessor(str(local_processor))
        return original_create_qwen2_tokenizer(pretrained_model_name, config_variant, *args, **kwargs)

    reasoner_defaults.create_qwen2_tokenizer_with_download = create_qwen2_tokenizer_local

    # Keep the visual branch enabled for checkpoints that actually carry the
    # VLM tower.  The converted Super DCP is a text-conditioned generation
    # checkpoint and omits those state-dict keys; matching that shape avoids a
    # DCP load failure while leaving HF/Nano checkpoints unchanged.
    original_create_vlm_config = reasoner_defaults.create_vlm_config

    def create_vlm_config_local(base_config, **overrides):
        overrides.setdefault("include_visual", _checkpoint_has_visual_weights())
        return original_create_vlm_config(base_config, **overrides)

    reasoner_defaults.create_vlm_config = create_vlm_config_local
    print(f">>> Generator processor local path: {local_processor}", flush=True)


def patch_local_guardrail_assets() -> None:
    """Route only Guardrail weights to local directories when requested."""
    guardrail_path = _local_directory("COSMOS_GUARDRAIL1_PATH")
    qwen3guard_path = _local_directory("COSMOS_QWEN3GUARD_PATH", "COSMOS_HCU_QWEN3GUARD_DIR")

    if guardrail_path is None and qwen3guard_path is None:
        return

    if guardrail_path is not None:
        required_paths = (
            guardrail_path / "blocklist",
            guardrail_path / "face_blur_filter" / "Resnet50_Final.pth",
        )
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "COSMOS_GUARDRAIL1_PATH is not a compatible Guardrail1 repository; missing: "
                + ", ".join(missing_paths)
            )

        from cosmos_framework.auxiliary.guardrail.common.core import GUARDRAIL1_CHECKPOINT

        # This is the same local-path mechanism used by the existing HCU
        # compatibility launcher. ``download()`` now returns this directory
        # and never invokes the Hugging Face CLI.
        GUARDRAIL1_CHECKPOINT._path = str(guardrail_path)
        print(f">>> Guardrail1 local path: {guardrail_path}", flush=True)

    if qwen3guard_path is not None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from cosmos_framework.auxiliary.guardrail.qwen3guard import qwen3guard

        def qwen3guard_init_local(self, offload_model_to_cpu: bool = True) -> None:
            self.offload_model = offload_model_to_cpu
            self.dtype = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(str(qwen3guard_path), local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(str(qwen3guard_path), local_files_only=True)
            device = "cpu" if offload_model_to_cpu else "cuda"
            self.model = self.model.to(device, dtype=self.dtype).eval()

        qwen3guard.Qwen3Guard.__init__ = qwen3guard_init_local
        print(f">>> Qwen3Guard local path: {qwen3guard_path}", flush=True)


def main() -> None:
    patch_hcu_python_compat()
    patch_local_generator_assets()
    patch_local_guardrail_assets()
    runpy.run_module("cosmos_framework.scripts.inference", run_name="__main__")


if __name__ == "__main__":
    main()
