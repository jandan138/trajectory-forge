"""Four-engine zero-cheat rollout workers and evaluator for distillation_v04."""

from .common_v04 import (
    ARM_BASE_QUAT_WXYZ,
    LEFT_ARM_BASE_POS,
    RIGHT_ARM_BASE_POS,
    V04_TASKS,
    bimanual_pad_world_poses,
    build_task_rigid_bodies_spec,
    evaluate_v04_task_outcome,
    load_v04_task_context,
    save_v04_episode_bundle,
)
from .runner_v04 import ENGINE_EXECUTABLES, run_v04_engine_batch

__all__ = [
    "ARM_BASE_QUAT_WXYZ",
    "ENGINE_EXECUTABLES",
    "LEFT_ARM_BASE_POS",
    "RIGHT_ARM_BASE_POS",
    "V04_TASKS",
    "bimanual_pad_world_poses",
    "build_task_rigid_bodies_spec",
    "evaluate_v04_task_outcome",
    "load_v04_task_context",
    "run_v04_engine_batch",
    "save_v04_episode_bundle",
]

