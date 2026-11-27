from __future__ import annotations

import ast

from trajectory_forge.core.contracts import ControllerSafetyError


FORBIDDEN_ROOT_MODULES: frozenset[str] = frozenset(
    {
        "builtins",
        "ctypes",
        "http",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "pty",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "threading",
        "urllib",
    }
)

FORBIDDEN_CALL_NAMES: frozenset[str] = frozenset(
    {"__import__", "compile", "eval", "exec", "getattr", "open", "setattr", "delattr"}
)

ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset({"math", "numpy", "trajectory_forge", "typing"})


def validate_controller_source(code: str, *, required_fn: str = "solve") -> None:
    """Validate an agent-authored Python controller script via AST inspection."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ControllerSafetyError(f"Controller syntax error: {exc}") from exc

    has_entry_fn = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == required_fn:
            if len(node.args.args) >= 1:
                has_entry_fn = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_ROOT_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                    raise ControllerSafetyError(
                        f"Disallowed import in controller: {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root in FORBIDDEN_ROOT_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                raise ControllerSafetyError(
                    f"Disallowed import-from in controller: {mod!r}"
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                raise ControllerSafetyError(
                    f"Disallowed builtin call in controller: {node.func.id!r}"
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen"}:
                raise ControllerSafetyError(
                    f"Disallowed attribute call in controller: {node.func.attr!r}"
                )

    if not has_entry_fn:
        raise ControllerSafetyError(f"Controller must define top-level `def {required_fn}(...):`")
