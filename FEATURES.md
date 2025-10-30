# Feature Enhancements

The following roadmap outlines high-impact additions that would extend the Fast Marching Tree stack. Each item focuses on research utility, reproducibility, or developer ergonomics.

## Planner and Mapping

- **Adaptive sampling policies:** Introduce obstacle-aware sampling densities that increase resolution near narrow passages while reducing samples in open space.
- **Dynamic obstacle support:** Extend the map loader to accept time-varying occupancy grids and integrate obstacle prediction for moving agents.

## Control and Simulation

- **Nonlinear MPC module:** Add a CasADi-backed nonlinear MPC option to test higher-fidelity vehicle dynamics while preserving the existing linear MPC as a fallback.
- **Robust disturbance models:** Inject configurable wind or slope disturbances into the simulator to benchmark controller resilience.

## Tooling and Visualization

- **Interactive scenario editor:** Build a lightweight GUI or notebook widget that edits YAML configurations and previews inflated maps in real time.
- **Automated regression dashboards:** Export planner and controller metrics to an HTML report with embedded plots for continuous integration pipelines.
