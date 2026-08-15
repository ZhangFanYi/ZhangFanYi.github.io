# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Apex optimizer adapters for FSDP2/DTensor parameters.

Apex's stock ``FusedAdam`` sends ``Parameter`` and ``grad`` objects directly
to ``multi_tensor_adam``.  FSDP2 exposes DTensor-backed parameters, while the
HCU Apex kernel requires regular local tensors.  The adapter keeps the
optimizer's parameter identity (needed by FSDP/DCP) and unwraps only the
multi-tensor kernel arguments.
"""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.utils.misc import get_local_tensor_if_DTensor


def build_apex_fused_adam(params: Any, **optimizer_kwargs: Any) -> torch.optim.Optimizer:
    """Build Apex FusedAdam with DTensor-safe multi-tensor inputs.

    The Apex import is intentionally delayed so environments without Apex can
    still import the normal optimizer registry and use native AdamW/FusedAdam.
    """

    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.optimizers import FusedAdam as ApexFusedAdam

    class DTensorCompatibleApexFusedAdam(ApexFusedAdam):
        """Apex FusedAdam whose kernel lists contain local tensors only."""

        @staticmethod
        def _local(tensor: torch.Tensor) -> torch.Tensor:
            return get_local_tensor_if_DTensor(tensor)

        def zero_grad(self, set_to_none: bool = True) -> None:
            """Match the torch Optimizer API used by the Cosmos trainer."""
            if set_to_none:
                for group in self.param_groups:
                    for param in group["params"]:
                        param.grad = None
            else:
                for group in self.param_groups:
                    for param in group["params"]:
                        if param.grad is None:
                            continue
                        local_grad = self._local(param.grad)
                        if local_grad.grad_fn is not None:
                            local_grad.detach_()
                        else:
                            local_grad.requires_grad_(False)
                        local_grad.zero_()

        def step(
            self,
            closure: Any = None,
            grads: Any = None,
            output_params: Any = None,
            scale: Any = None,
            grad_norms: Any = None,
            grad_scaler: Any = None,
        ) -> Any:
            if any(value is not None for value in [grads, output_params, scale, grad_norms]):
                raise RuntimeError(
                    "FusedAdam has been updated. Simply initialize it identically to torch.optim.Adam "
                    "and call step() with no arguments."
                )

            loss = closure() if closure is not None else None

            for group, group_master in zip(self.param_groups, self.param_groups_master):
                if len(group["params"]) == 0:
                    continue

                first_param = self._local(group["params"][0])
                device = first_param.device
                bias_correction = 1 if group["bias_correction"] else 0
                beta1, beta2 = group["betas"]

                if "step" in group:
                    group["step"] += 1 if not self.capturable else (self._dummy_overflow_buf != 1).to(torch.int)
                else:
                    group["step"] = (
                        1
                        if not self.capturable
                        else torch.tensor([1], dtype=torch.int, device=device)
                    )

                g_16: list[torch.Tensor] = []
                p_16: list[torch.Tensor] = []
                m_16: list[torch.Tensor] = []
                v_16: list[torch.Tensor] = []
                g_bf: list[torch.Tensor] = []
                p_bf: list[torch.Tensor] = []
                m_bf: list[torch.Tensor] = []
                v_bf: list[torch.Tensor] = []
                g_32: list[torch.Tensor] = []
                p_32: list[torch.Tensor] = []
                m_32: list[torch.Tensor] = []
                v_32: list[torch.Tensor] = []
                p_16_master: list[torch.Tensor] = []
                p_32_master: list[torch.Tensor] = []

                for param, param_master in zip(group["params"], group_master["params"]):
                    if param.grad is None:
                        continue

                    local_param = self._local(param)
                    local_grad = self._local(param.grad)
                    if local_grad.is_sparse:
                        raise RuntimeError(
                            "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    state = self.state[param]
                    if len(state) == 0:
                        # Keep optimizer moments DTensor-aware for FSDP2/DCP.
                        # Only the arguments handed to Apex's CUDA kernel are
                        # unwrapped below; storing local tensors here would
                        # make optimizer state appear replicated to FSDP2.
                        state["exp_avg"] = torch.zeros_like(param).float()
                        state["exp_avg_sq"] = torch.zeros_like(param).float()

                    exp_avg = self._local(state["exp_avg"])
                    exp_avg_sq = self._local(state["exp_avg_sq"])

                    if param.dtype == torch.float16:
                        if self.master_weights:
                            p_16_master.append(self._local(param_master.data))
                        g_16.append(local_grad.data)
                        p_16.append(local_param.data)
                        m_16.append(exp_avg)
                        v_16.append(exp_avg_sq)
                    elif param.dtype == torch.bfloat16:
                        g_bf.append(local_grad)
                        p_bf.append(local_param)
                        m_bf.append(exp_avg)
                        v_bf.append(exp_avg_sq)
                    elif param.dtype == torch.float32:
                        if self.master_weights:
                            p_32_master.append(self._local(param_master.data))
                        g_32.append(local_grad.data)
                        p_32.append(local_param.data)
                        m_32.append(exp_avg)
                        v_32.append(exp_avg_sq)
                    else:
                        raise RuntimeError("FusedAdam only supports fp16, bf16 and fp32.")

                if self.capturable:
                    found_inf = (
                        grad_scaler._check_inf_per_device(self)[device]
                        if grad_scaler is not None
                        else torch.zeros((1,), device=device)
                    )
                    self._dummy_overflow_buf.copy_(found_inf)
                    scale_value = grad_scaler._get_scale_async() if grad_scaler else torch.ones((1,), device=device)
                    inv_scale = scale_value.double().reciprocal().float()

                    if g_16:
                        multi_tensor_applier(
                            self.multi_tensor_adam_capturable_master if self.master_weights else self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_16, p_16, m_16, v_16, p_16_master] if self.master_weights else [g_16, p_16, m_16, v_16],
                            group["lr"], beta1, beta2, group["eps"], group["step"],
                            self.adam_w_mode, bias_correction, group["weight_decay"], inv_scale,
                        )
                    if g_bf:
                        multi_tensor_applier(
                            self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_bf, p_bf, m_bf, v_bf],
                            group["lr"], beta1, beta2, group["eps"], group["step"],
                            self.adam_w_mode, bias_correction, group["weight_decay"], inv_scale,
                        )
                    if g_32:
                        multi_tensor_applier(
                            self.multi_tensor_adam_capturable_master if self.master_weights else self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_32, p_32, m_32, v_32, p_32_master] if self.master_weights else [g_32, p_32, m_32, v_32],
                            group["lr"], beta1, beta2, group["eps"], group["step"],
                            self.adam_w_mode, bias_correction, group["weight_decay"], inv_scale,
                        )
                else:
                    for grads_list, params_list, exp_avg_list, exp_avg_sq_list in (
                        (g_16, p_16, m_16, v_16),
                        (g_bf, p_bf, m_bf, v_bf),
                        (g_32, p_32, m_32, v_32),
                    ):
                        if grads_list:
                            multi_tensor_applier(
                                self.multi_tensor_adam,
                                self._dummy_overflow_buf,
                                [grads_list, params_list, exp_avg_list, exp_avg_sq_list],
                                group["lr"], beta1, beta2, group["eps"], group["step"],
                                self.adam_w_mode, bias_correction, group["weight_decay"],
                            )

            return loss

    return DTensorCompatibleApexFusedAdam(params, **optimizer_kwargs)
