from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
PURE_MODULE_ROOTS = (
    REPO_ROOT / "src" / "trajectory_forge" / "core",
    REPO_ROOT / "src" / "trajectory_forge" / "intake",
    REPO_ROOT / "src" / "trajectory_forge" / "embodiment",
    REPO_ROOT / "src" / "trajectory_forge" / "sdk",
    REPO_ROOT / "src" / "trajectory_forge" / "worker",
    REPO_ROOT / "src" / "trajectory_forge" / "trajectory",
)
FORBIDDEN_IMPORT_MARKERS = (
    "import pxr",
    "from pxr",
    "import omni",
    "from omni",
    "import isaacsim",
    "from isaacsim",
    "import genmanip",
    "from genmanip",
    "import curobo",
    "from curobo",
    "import mujoco",
    "from mujoco",
    "import torch",
    "from torch",
)
FORBIDDEN_IDENTITY_RE = re.compile(r"\b(astra)\b", re.IGNORECASE)


def test_pure_package_layers_do_not_import_heavy_simulator_stacks() -> None:
    violations: list[str] = []
    for root in PURE_MODULE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for marker in FORBIDDEN_IMPORT_MARKERS:
                if marker in source:
                    violations.append(f"{path.relative_to(REPO_ROOT)} contains {marker!r}")
    assert violations == []


def test_no_external_model_or_lab_identity_in_core_contracts() -> None:
    violations: list[str] = []
    for root in PURE_MODULE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if "lab" in path.stem.lower() or FORBIDDEN_IDENTITY_RE.search(path.stem):
                violations.append(f"forbidden module name: {rel}")
            source = path.read_text(encoding="utf-8")
            if FORBIDDEN_IDENTITY_RE.search(source):
                violations.append(f"forbidden external model token in {rel}")
    assert violations == []
