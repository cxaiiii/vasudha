from typing import Dict, Any

from .rms_norm import fused_rms_norm, VasudhaRMSNorm, HAS_TRITON as RMS_HAS_TRITON
from .swiglu import fused_swiglu, fused_swiglu_inplace, HAS_TRITON as SWIGLU_HAS_TRITON
from .cross_entropy import vasudha_cross_entropy, VasudhaCrossEntropyLoss, HAS_TRITON as CE_HAS_TRITON
from .linear_attn import gla_forward, HAS_TRITON as GLA_HAS_TRITON
from .moe_dispatch import moe_dispatch, HAS_TRITON as MOE_DISPATCH_HAS_TRITON
from .moe_gather import moe_gather, HAS_TRITON as MOE_GATHER_HAS_TRITON

__all__ = [
    "fused_rms_norm",
    "VasudhaRMSNorm",
    "fused_swiglu",
    "fused_swiglu_inplace",
    "vasudha_cross_entropy",
    "VasudhaCrossEntropyLoss",
    "gla_forward",
    "moe_dispatch",
    "moe_gather",
    "get_kernel_status",
]

def get_kernel_status() -> Dict[str, str]:
    """
    Returns a dictionary mapping kernel names to their current execution backend
    ('triton' or 'pytorch_fallback').
    """
    return {
        "rms_norm": "triton" if RMS_HAS_TRITON else "pytorch_fallback",
        "swiglu": "triton" if SWIGLU_HAS_TRITON else "pytorch_fallback",
        "cross_entropy": "triton" if CE_HAS_TRITON else "pytorch_fallback",
        "linear_attn": "triton" if GLA_HAS_TRITON else "pytorch_fallback",
        "moe_dispatch": "triton" if MOE_DISPATCH_HAS_TRITON else "pytorch_fallback",
        "moe_gather": "triton" if MOE_GATHER_HAS_TRITON else "pytorch_fallback",
    }
