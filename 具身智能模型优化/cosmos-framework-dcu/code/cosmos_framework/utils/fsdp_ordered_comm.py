# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ordered FSDP collective submission streams.

NCCL-compatible backends require collectives for one communicator to be
submitted in the same global order. FSDP2 intentionally uses separate streams
for local copy/compute overlap, so the communication call itself is handed off
to one stream per process group while the surrounding local work stays on the
FSDP-managed streams.
"""

import os
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate


_streams: dict[tuple[str, int, dist.ProcessGroup], torch.Stream] = {}
_streams_lock = threading.RLock()


def _group_key(
    group: dist.ProcessGroup, device: torch.device
) -> tuple[str, int, dist.ProcessGroup]:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return device.type, device_index, group


def _get_group_stream(group: dist.ProcessGroup, device: torch.device) -> torch.Stream:
    key = _group_key(group, device)
    with _streams_lock:
        stream = _streams.get(key)
        if stream is None:
            with torch.cuda.device(key[1]):
                stream = torch.cuda.Stream(device=key[1])
            _streams[key] = stream
        return stream


@contextmanager
def _ordered_group_stream(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    wait_for_completion: bool = True,
) -> Iterator[None]:
    """Run one collective on the process group's canonical submission stream.

    ``wait_for_completion=False`` is intentionally opt-in.  It is safe only
    for an asynchronous collective whose caller later waits on the returned
    work before consuming the output.  Keeping the default synchronous stream
    wait preserves the original ordering contract for all existing callers.
    """
    caller_stream = torch.cuda.current_stream(device)
    collective_stream = _get_group_stream(group, device)
    with _streams_lock:
        collective_stream.wait_stream(caller_stream)
        with torch.cuda.stream(collective_stream):
            yield
        if wait_for_completion:
            caller_stream.wait_stream(collective_stream)


def _async_all_gather_enabled() -> bool:
    """Return whether FSDP may defer the caller-stream completion wait.

    The environment variable is deliberately opt-in because a generic
    prefetch is not beneficial when the next parameter is consumed
    immediately.  FSDP2's all-gather copy-out path owns the returned Work and
    must wait before reading the output tensor.
    """
    return os.getenv("COSMOS_FSDP_ASYNC_ALL_GATHER", "0").strip().lower() in {"1", "true", "yes"}


class OrderedAllGather:
    """FSDP2 all-gather implementation with per-process-group stream ordering."""

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        defer_caller_wait = async_op and _async_all_gather_enabled()
        with _ordered_group_stream(
            group,
            input_tensor.device,
            wait_for_completion=not defer_caller_wait,
        ):
            return dist.all_gather_into_tensor(
                output_tensor,
                input_tensor,
                group=group,
                async_op=async_op,
            )


class OrderedReduceScatter:
    """FSDP2 reduce-scatter implementation with per-process-group stream ordering."""

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: dist.ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        with _ordered_group_stream(group, input_tensor.device):
            return dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                group=group,
                op=op,
                async_op=async_op,
            )


def ordered_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    """Submit an all-reduce on the process group's canonical stream."""
    if group is None:
        group = dist.group.WORLD
    with _ordered_group_stream(group, tensor.device):
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def materialize_dtensor_norm(norm: DTensor) -> torch.Tensor:
    """Reduce a DTensor norm while preserving per-communicator stream order.

    A norm DTensor contains only Replicate and norm-specific Partial placements.
    Reducing each partial placement explicitly is equivalent to full_tensor(),
    while exposing the process-group boundary needed to choose the same
    submission stream used by FSDP collectives.
    """
    local_norm = norm.to_local()
    mesh = norm.device_mesh
    for mesh_dim, placement in enumerate(norm.placements):
        if isinstance(placement, Replicate):
            continue
        if not isinstance(placement, Partial):
            raise RuntimeError(f"Unexpected gradient norm DTensor placement: {placement}")
        group = mesh.get_group(mesh_dim)
        with _ordered_group_stream(group, local_norm.device):
            local_norm = placement._reduce_value(local_norm, mesh, mesh_dim)
    return local_norm
