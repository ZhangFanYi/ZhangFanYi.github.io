# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a Hugging Face model to a DCP checkpoint."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import math
import os
import shutil
from typing import Annotated

import pydantic
import torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS


_AVAE_REGISTRY_URI = "s3://bucket/pretrained/tokenizers/audio/avae"


def _redirect_avae_to_local(hf_path):
    """Point the AVAE registry entry at hf_path/sound_tokenizer/.

    Pre-seeds the CheckpointDirHf._path cache so the registered AVAE checkpoint
    resolves to the local sibling directory instead of fetching the pinned
    revision of nvidia/Cosmos3-Nano from the HF Hub during hydra instantiation.
    """
    sound_tokenizer_dir = hf_path / "sound_tokenizer"
    if not sound_tokenizer_dir.is_dir():
        return
    register_checkpoints()
    avae = _CHECKPOINTS.get(_AVAE_REGISTRY_URI)
    if avae is not None:
        avae.hf._path = str(sound_tokenizer_dir)


def _redirect_vlm_processor_to_local(config: Cosmos3OmniConfig, hf_path):
    """Use processor files bundled with a local Cosmos3 HF checkpoint."""
    tokenizer = config.model.get("config", {}).get("vlm_config", {}).get("tokenizer", {})
    if not isinstance(tokenizer, dict) or not (hf_path / "tokenizer.json").is_file():
        return
    tokenizer["local_processor_dir"] = str(hf_path)
    tokenizer["config_variant"] = "hf"


def _redirect_video_vae_to_local(config: Cosmos3OmniConfig):
    """Use a local Wan2.2 VAE when WAN_VAE_PATH is provided."""
    vae_path = os.environ.get("WAN_VAE_PATH")
    if not vae_path:
        return
    if not os.path.isfile(vae_path):
        raise FileNotFoundError(f"WAN_VAE_PATH not found: {vae_path}")
    tokenizer = config.model.get("config", {}).get("tokenizer", {})
    tokenizer["vae_path"] = vae_path
    tokenizer["bucket_name"] = ""
    tokenizer["object_store_credential_path_pretrained"] = ""


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Hugging Face checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output DCP checkpoint directory."""


def convert_model_to_dcp(args: Args):
    print("Loading model...")
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    hf_path = checkpoint_config.download_checkpoint()
    _redirect_avae_to_local(hf_path)
    model_dict = checkpoint_config.load_model_config_dict()
    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    _redirect_vlm_processor_to_local(hf_config, hf_path)
    _redirect_video_vae_to_local(hf_config)
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=hf_config)
    state_dict = get_model_state_dict(hf_model.model)

    # Match transformers default max shard size = 5GB.
    max_shard_size = 5 * 1024**3
    model_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
    thread_count = math.ceil(model_size / max_shard_size)

    print("Saving model...")
    storage_writer = FileSystemWriter(args.output_path / "model", thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    shutil.copy(hf_path / "checkpoint.json", args.output_path / "checkpoint.json")
    hf_config.save_pretrained(args.output_path / "model")
    print(f"Saved checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_dcp(args)


if __name__ == "__main__":
    main()
