"""Pluggable Trajectory Generation Paradigm Plugins (Microkernel v2 Upper Layer)."""
from trajectory_forge.producers.base import ProducerRegistry, ensure_adapter_capabilities
from trajectory_forge.producers.code_to_control import CodeToControlProducer
from trajectory_forge.producers.direct_control import DirectControlProducer, DirectControlSession
from trajectory_forge.producers.rl_policy import RLPolicyProducer

DEFAULT_PRODUCER_REGISTRY = ProducerRegistry()
DEFAULT_PRODUCER_REGISTRY.register(CodeToControlProducer())
DEFAULT_PRODUCER_REGISTRY.register(DirectControlProducer())
DEFAULT_PRODUCER_REGISTRY.register(RLPolicyProducer())

__all__ = [
    "DEFAULT_PRODUCER_REGISTRY",
    "CodeToControlProducer",
    "DirectControlProducer",
    "DirectControlSession",
    "ProducerRegistry",
    "RLPolicyProducer",
    "ensure_adapter_capabilities",
]
