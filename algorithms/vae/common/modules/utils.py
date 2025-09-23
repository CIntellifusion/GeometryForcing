from typing import Literal, List
from functools import partial
import importlib

Module = Literal[
    "",
    "Conv2d",
    "PaddedConv3D",
    "AttnBlock",
    "AttnBlock3D",
    "ResnetBlock3D",
    "Upsample",
    "Downsample",
    "SpatialUpsample2x",
    "SpatialDownsample2x",
    "Spatial2xTime2x3DUpsample",
    "Spatial2xTime2x3DDownsample",
]

MODULES_3D: List[Module] = [
    "PaddedConv3D",
    "AttnBlock3D",
    "ResnetBlock3D",
    "SpatialUpsample2x",
    "SpatialDownsample2x",
    "Spatial2xTime2x3DUpsample",
    "Spatial2xTime2x3DDownsample",
]

MODULES_BASE = "algorithms.vae.common.modules"


# Returns the module class given the module name.
def resolve_str_to_module(name: Module, is_causal: bool) -> type:
    if name == "":
        raise ValueError("Empty string is not a valid module name.")
    module = importlib.import_module(MODULES_BASE)
    module_cls = getattr(module, name)
    if name in MODULES_3D:
        module_cls = partial(module_cls, is_causal=is_causal)
    return module_cls
