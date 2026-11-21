"""Resolve newer library symbols while running a prebuilt SpecForge eager model.

The unused FlexAttention and HF model-construction/restore functions raise if
called. The probe uses eager attention, real torch autograd, prebuilt weights,
and explicit visibility tensors. No edits to installed Python packages.
"""

import sys
import types


def unavailable(*args, **kwargs):
    raise RuntimeError("The legacy import adapter does not implement this operation")


def install():
    import torch
    import transformers
    import transformers.activations as activations
    import transformers.modeling_utils as modeling_utils

    if not hasattr(activations, "SiLUActivation"):

        class SiLUActivation(torch.nn.Module):
            def forward(self, hidden):
                return torch.nn.functional.silu(hidden)

        activations.SiLUActivation = SiLUActivation

    if not hasattr(transformers, "PreTrainedConfig"):
        transformers.PreTrainedConfig = transformers.PretrainedConfig
    if not hasattr(transformers, "initialization"):
        transformers.initialization = torch.nn.init
    for name in ("get_device", "get_torch_context_manager_or_global_device"):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, unavailable)
    try:
        import torch.nn.attention.flex_attention
    except ImportError:
        attention = types.ModuleType("torch.nn.attention")
        flex = types.ModuleType("torch.nn.attention.flex_attention")
        for name in (
            "create_mask",
            "AuxRequest",
            "create_block_mask",
            "flex_attention",
        ):
            setattr(flex, name, unavailable)
        attention.flex_attention = flex
        sys.modules[attention.__name__] = attention
        sys.modules[flex.__name__] = flex
