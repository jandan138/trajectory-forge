"""Simulator adapter plugins isolated from pure microkernel layers."""
from trajectory_forge.adapters.base import SimAdapterRegistry
from trajectory_forge.adapters.isaac_genmanip.sim_adapter import IsaacGenManipSimAdapter
from trajectory_forge.adapters.newton.newton_adapter import NewtonSimAdapter

DEFAULT_SIM_ADAPTER_REGISTRY = SimAdapterRegistry()
DEFAULT_SIM_ADAPTER_REGISTRY.register(IsaacGenManipSimAdapter())
DEFAULT_SIM_ADAPTER_REGISTRY.register(NewtonSimAdapter())

__all__ = [
    "DEFAULT_SIM_ADAPTER_REGISTRY",
    "IsaacGenManipSimAdapter",
    "NewtonSimAdapter",
    "SimAdapterRegistry",
]
