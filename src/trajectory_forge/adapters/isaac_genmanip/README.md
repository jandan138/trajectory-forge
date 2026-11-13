# Isaac Sim 4.1 + GenManip Adapter Boundary

This adapter is the **only** location permitted to interact with `isaacsim`, `omni.*`, `pxr`, `GenManip-Sim`, or GPU motion planners (`CuRobo`), and it runs inside the RTX 4090 host's `embodied-eval-os-isaacsim41-py310` environment via a subprocess boundary.

Core pure layers (`core`, `intake`, `embodiment`, `sdk`, `worker`, `trajectory`) stay 100% simulator-free so cloud agents in Google AI Ultra can parse briefs, verify math, and run `DryRunSimBackend` preflight without Omniverse dependencies.
