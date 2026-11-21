"""Pure sensorimotor controllers for M3 deformable manipulation tasks."""

from trajectory_forge.core.contracts import ScenarioIntakeBrief
from examples.controllers.deformable_v02.coffee_pastry_crush_counterfactual import (
    CoffeePastryCrushController,
)
from examples.controllers.deformable_v02.coffee_pastry_gentle_controller import (
    CoffeePastryGentleController,
)
from examples.controllers.deformable_v02.office_cloth_tear_counterfactual import (
    OfficeClothTearController,
)
from examples.controllers.deformable_v02.office_cloth_wipe_gentle_controller import (
    OfficeClothWipeGentleController,
)


def build_controller_for_mode(task_id: str, mode: str, brief: ScenarioIntakeBrief):
    if task_id == "coffee_bar_fragile_pastry_plate":
        if mode == "gentle":
            return CoffeePastryGentleController(brief)
        if mode == "crush":
            return CoffeePastryCrushController(brief)
    if task_id == "office_desk_wiping_cloth_clean_and_park":
        if mode == "gentle":
            return OfficeClothWipeGentleController(brief)
        if mode == "tear":
            return OfficeClothTearController(brief)
    raise ValueError(f"Unsupported task/mode combination: task_id={task_id!r}, mode={mode!r}")


__all__ = [
    "CoffeePastryCrushController",
    "CoffeePastryGentleController",
    "OfficeClothTearController",
    "OfficeClothWipeGentleController",
    "build_controller_for_mode",
]

