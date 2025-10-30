# Introduction

## Overview

This repository implements a **Fast Marching Tree (FMT)**–based autonomy stack for ground vehicle navigation in structured environments represented by occupancy grids. The system combines **sampling-based global planning**, **model-predictive control**, **deterministic simulation**, and **scientific visualization** into a unified framework for research-grade experimentation. Its primary design goals are **reproducibility**, **scientific instrumentation**, and **modularity**, allowing consistent comparison of experiments and precise inspection of algorithmic behavior.

FMT planning offers the ability to compute high-quality motion trajectories through cluttered environments while preserving computational efficiency. The surrounding stack ensures that each subsystem—from mapping to control—operates on deterministic data, yielding identical results across repeated runs.

---

## 1. System Composition

The architecture integrates four interdependent scientific pillars:

1. **Mapping**
   Constructs deterministic occupancy grids from images or specifications.

   * Performs **OpenCV-based obstacle inflation** to impose clearance constraints.
   * Annotates each grid with **metric metadata** (resolution, scale, and origin) to ensure consistent spatial interpretation throughout the pipeline.

2. **Planning**
   Implements a **goal-biased Fast Marching Tree (FMT)** planner to find feasible paths in the free configuration space $\mathcal{X}_{\mathrm{free}}$.

   * Generates a sparse graph of feasible connections respecting inflated safety corridors.
   * Produces detailed **exploration logs** capturing sampling, edge validation, and cost propagation.
   * Applies **spline smoothing** and **corridor validation** to generate curvature-continuous paths.

3. **Tracking**
   Uses a **Model Predictive Controller (MPC)** formulated in **cvxpy**.

   * Linearizes vehicle dynamics along the planned reference.
   * Solves a constrained quadratic program (QP) at every timestep, yielding optimal acceleration and steering rate inputs.
   * Outputs both feedforward and feedback actions, as well as predicted state horizons for interpretability.

4. **Simulation & Visualization**
   Provides a **deterministic kinematic bicycle simulator** and an extensive visualization layer.

   * The simulator integrates the dynamics with fixed time discretization, ensuring repeatability.
   * Visual outputs include static dashboards, animated trajectories, and planner debug views.
   * The result is a complete end-to-end diagnostic system capable of both numerical analysis and graphical validation.

Scenario configurations are defined as YAML manifests (located in `configs/`), which specify the **map**, **vehicle**, **planner**, and **visualization** parameters. These configurations are executed by the **command-line interface** in `src/cli`, enabling selective subsystem execution or full end-to-end experiments.

---

## 2. Theoretical Rationale: Combining FMT and MPC

### 2.1. FMT: Global Sampling-Based Optimization

The Fast Marching Tree (FMT) algorithm constructs a sparse, directed acyclic graph of feasible motions by combining **sampling-based exploration** with **dynamic programming principles**.
Given a set of free-space samples $\mathcal{X}_{\mathrm{free}} = \{ x_i \}$, the planner seeks a path

$$
\xi = (x_0, x_1, \dots, x_N)
$$

minimizing the geometric cost:

$$
J(\xi) = \sum_{k=0}^{N-1} \lVert x_{k+1} - x_k \rVert_2,
$$

subject to all edges lying within the inflated safety corridor.

FMT achieves **asymptotic optimality** while maintaining computational complexity comparable to Probabilistic Roadmaps (PRM). It incrementally expands a frontier of optimal partial paths and connects only nearby feasible nodes, drastically reducing redundant collision checks.

### 2.2. MPC: Local Dynamic Optimization

While FMT provides a collision-free global reference, it does not explicitly enforce dynamic constraints of the vehicle. To ensure **feasible and stable motion**, a Model Predictive Controller refines this reference online.

At each control step, MPC solves the convex quadratic optimization problem:

$$
\begin{aligned}
\min_{u_{0:H-1}} \quad & \sum_{k=0}^{H} \lVert x_k - x_k^{\mathrm{ref}} \rVert_Q^2 + \sum_{k=0}^{H-1} \lVert u_k \rVert_R^2 \\
\text{s.t.} \quad & x_{k+1} = f(x_k, u_k),
\end{aligned}
$$

subject to the discretized vehicle model and actuator constraints, where $f(\cdot)$ denotes the discretized bicycle dynamics used throughout the stack.
Here:

* $H$: prediction horizon length.
* $x_k^{\mathrm{ref}}$: spline-sampled reference states along the FMT path.
* $Q$, $R$: positive-definite weighting matrices penalizing tracking error and control effort.

The MPC adjusts the local vehicle behavior to remain dynamically feasible and robust to disturbances while adhering to the global plan.

### 2.3. Coupled Hierarchy

The joint operation of FMT and MPC yields a **hierarchical planning and control system**:

* FMT defines the **global corridor** with guaranteed collision-free geometry.
* MPC enforces **dynamic feasibility** and real-time correction.
* The deterministic simulator ensures **exact repeatability** for all combinations of seeds and parameters.

Together, they enable stable trajectory generation and control that can be replicated across experiments, ensuring reproducibility for research and benchmarking.

---

## 3. Reproducibility and Instrumentation

The framework is built for **scientific reproducibility** and detailed instrumentation.
Each component emits structured logs containing:

* Planner statistics (sample counts, connection radius, cost metrics).
* Controller diagnostics (solver convergence, constraint activations, error trajectories).
* Simulator states (pose, curvature, slip metrics).
* Visualization artifacts (map overlays, vehicle poses, predicted horizons).

All numerical operations, from random sampling to time integration, are deterministic and seed-controlled, ensuring **identical results across repeated runs**.

---

## 4. Usage and Experimental Workflow

The experiment workflow proceeds as follows:

1. **Scenario Definition:**
   Create or modify a YAML file specifying the map, vehicle model, planner parameters, and visualization options.

2. **Execution:**
   Run the desired subsystem or full pipeline via the `src/cli` command-line interface.

3. **Computation:**
   The stack executes the mapping, planning, smoothing, control, and simulation modules sequentially, generating a complete trajectory and log.

4. **Visualization and Analysis:**
   Visual outputs and log files provide both static and dynamic insight into the experiment, enabling quantitative performance evaluation.

This modular design supports reproducible studies, parameter sweeps, and comparative benchmarking of algorithmic variants under identical environmental conditions.

---

## 5. Summary

The repository delivers a **scientifically structured**, **deterministic**, and **extensible** autonomy stack integrating:

* **Global motion planning** via Fast Marching Tree (FMT).
* **Dynamic control optimization** via Model Predictive Control (MPC).
* **Deterministic simulation** ensuring bitwise repeatability.
* **Comprehensive visualization and logging** for analytical traceability.

By uniting global optimality with local dynamic feasibility, the framework enables rigorous experimentation in autonomous ground vehicle motion planning and control, bridging theoretical soundness with reproducible practical execution.
