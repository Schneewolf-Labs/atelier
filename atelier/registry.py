"""String → class registry for adapters and losses.

Lets YAML configs reference adapters / losses by short name instead
of fully-qualified Python paths. Used by atelier.train CLI but also
fine to import programmatically.
"""
import importlib
from typing import Any

ADAPTERS: dict[str, str] = {
    "qwen_image": "atelier.adapters.qwen_image:QwenImageAdapter",
    "qwen_edit":  "atelier.adapters.qwen_edit:QwenEditAdapter",
    "sdxl":       "atelier.adapters.sdxl:SDXLAdapter",
}

LOSSES: dict[str, str] = {
    "flow_matching":  "atelier.losses.flow_matching:FlowMatchingLoss",
    "epsilon":        "atelier.losses.epsilon:EpsilonLoss",
    "diffusion_dpo":  "atelier.losses.diffusion_dpo:DiffusionDPOLoss",
    "diffusion_cpo":  "atelier.losses.diffusion_cpo:DiffusionCPOLoss",
    "diffusion_ipo":  "atelier.losses.diffusion_ipo:DiffusionIPOLoss",
    "diffusion_kto":  "atelier.losses.diffusion_kto:DiffusionKTOLoss",
    "diffusion_orpo": "atelier.losses.diffusion_orpo:DiffusionORPOLoss",
    "diffusion_simpo": "atelier.losses.diffusion_simpo:DiffusionSimPOLoss",
}


def _resolve(spec: str) -> Any:
    """'pkg.mod:Class' → the actual class object."""
    module_path, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"registry spec '{spec}' missing ':ClassName'")
    return getattr(importlib.import_module(module_path), name)


def get_adapter_class(name: str):
    """Resolve adapter short name to class. Accepts 'pkg.mod:Class' too."""
    spec = ADAPTERS.get(name, name)
    return _resolve(spec) if ":" in spec else _err("adapter", name, ADAPTERS)


def get_loss_class(name: str):
    """Resolve loss short name to class. Accepts 'pkg.mod:Class' too."""
    spec = LOSSES.get(name, name)
    return _resolve(spec) if ":" in spec else _err("loss", name, LOSSES)


def _err(kind: str, name: str, registry: dict[str, str]):
    known = ", ".join(sorted(registry))
    raise KeyError(f"unknown {kind} '{name}'. Known: {known}. "
                   f"Or pass a full 'package.module:ClassName' spec.")
