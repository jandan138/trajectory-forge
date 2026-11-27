"""Zero-cheat 4-engine rollout workers for v0.3 full-family deformable manipulation."""

from .common_v03 import (
    V03_TASKS,
    evaluate_v03_task_outcome,
    load_v03_proxy_and_brief,
)

__all__ = [
    "V03_TASKS",
    "evaluate_v03_task_outcome",
    "load_v03_proxy_and_brief",
]

