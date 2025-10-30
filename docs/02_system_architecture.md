# System Architecture

## Overview

The repository implements a modular and scientifically rigorous architecture for autonomous vehicle planning, control, and simulation. Each subsystem—mapping, planning, vehicle dynamics, optimization, visualization, and logging—resides in an isolated package with a well-defined interface. This compartmentalization ensures component-level replaceability, minimal coupling, and deterministic reproducibility across simulations.

The system is designed around **data-driven configuration**, **clear inter-module contracts**, and **reproducible computation**, enabling users to modify planners, vehicle models, or visualization components without affecting the rest of the framework.

---

## 1. Data Flow Overview

The overall pipeline follows a sequential yet modular execution chain, depicted below:

```mermaid
graph LR
    A[Scenario YAML] --> B[Configuration Loader]
    B --> C[Mapping]
    C --> D[FMT Planner]
    D --> E[Trajectory Smoother]
    E --> F[MPC Tracker]
    F --> G[Simulator]
    G --> H[Visualisation]
    H --> I[Artifacts]
```

Each stage transforms structured inputs into typed outputs that serve as inputs to subsequent modules. The interfaces are implemented via typed dataclasses and explicit contracts, ensuring static consistency and runtime integrity.

---

## 2. Core Modules

### 2.1. Configuration (`src/common/config.py`)

**Purpose:** Define and validate simulation and system parameters from YAML scenario files.

**Technical details:**

* Uses **Python dataclasses** with strict type annotations for every parameter.
* Validates field structures and permissible ranges.
* Seeds all pseudo-random number generators (`numpy`, `random`, etc.) to ensure deterministic sampling and reproducible randomization in planning.
* Provides bidirectional mapping between YAML schema and runtime configuration objects.
* Comprehensive docstrings establish explicit configuration contracts.

**Outcome:** Guarantees repeatable experiments, stable parameter propagation, and clear traceability between YAML scenarios and code execution.

---

### 2.2. Mapping (`src/mapping/`)

**Purpose:** Generate and preprocess the environment representation for the planner.

**Processing steps:**

1. Loads binary **occupancy grids** from deterministic specifications or bitmap images (PNG format).
2. Applies **morphological inflation** using OpenCV to construct inflated obstacle maps that maintain safety margins around physical obstacles.
3. Computes **Euclidean distance transforms**, yielding a scalar field $d_{\mathrm{obs}}(x)$ that encodes clearance from obstacles at every grid cell.

**Outputs:**

* Original occupancy map $\mathcal{O}$
* Inflated map $\mathcal{I}$
* Continuous distance field for clearance evaluation.

**Scientific role:** Enables consistent safety evaluation and real-time feasibility checking during planning and simulation.

---

### 2.3. Planning (`src/planning/fmt_planner.py`)

**Purpose:** Compute collision-free, near-optimal reference paths through the configuration space.

**Key features:**

* Implements **Fast Marching Tree (FMT*)** algorithm for asymptotically optimal path planning.
* Samples free configurations from $\mathcal{X}_{\mathrm{free}}$ with **goal biasing** to accelerate convergence.
* Connects nodes within a radius $r_n = \gamma (\log n / n)^{1/d}$, following theoretical optimality bounds.
* Validates edges using the inflated safety corridor and the precomputed distance transform.
* Records exploration statistics, including rejected edges, accepted nodes, and computational time.

**Scientific significance:** Balances computational efficiency with formal asymptotic guarantees, enabling rigorous comparison across runs.

---

### 2.4. Trajectory Smoothing (`src/tracking/mpc_tracker.py` and intermediary smoothing stage)

**Purpose:** Transform discrete waypoints from the planner into curvature-continuous, differentiable reference trajectories for MPC tracking.

**Methods:**

* Uses **arc-length reparameterization** to obtain uniformly spaced reference samples.
* Applies **centripetal Catmull–Rom interpolation** ($\alpha = 0.5$) to avoid overshooting near sharp turns.
* Validates post-smoothing path against the inflated corridor to preserve clearance constraints.

**Output:** A smooth trajectory with continuous position, curvature, and heading derivatives suitable for model-based control.

---

### 2.5. Tracking (`src/tracking/mpc_tracker.py`)

**Purpose:** Perform constrained optimal control using **Model Predictive Control (MPC)**.

**Implementation details:**

* Linearizes kinematic bicycle dynamics along the reference trajectory.
* Constructs discrete-time linear systems $x_{k+1} = A_k x_k + B_k u_k + c_k$.
* Solves a convex **Quadratic Program (QP)** formulated in CVXPY, incorporating:

  * State and control cost terms $\lVert x_k - x_k^{\mathrm{ref}} \rVert_Q^2$, $\lVert u_k - u_k^{\mathrm{ref}} \rVert_R^2$,
  * Velocity, steering, and actuation constraints.
* Executes in **receding-horizon fashion**, applying only the first optimal control at each iteration.

**Scientific outcome:** Realizes real-time, constraint-aware vehicle control with guaranteed convexity and stability margins.

---

### 2.6. Vehicle Model (`src/vehicle/bicycle.py`)

**Purpose:** Encapsulate the **kinematic bicycle dynamics** for both forward simulation and linearization.

**Functionalities:**

* Provides analytic state-update equations for position, heading, and velocity.
* Computes curvature $\kappa = \tan(\delta)/L$, slip ratio, and lateral acceleration.
* Supplies Jacobian matrices ( A, B ) for MPC linearization.
* Includes instrumentation to monitor numerical consistency of curvature and slip metrics.

**Scientific contribution:** Establishes a physically interpretable and mathematically tractable model for real-time control.

---

### 2.7. Simulation (`src/sim/simulator.py`)

**Purpose:** Execute a **closed-loop deterministic simulation** integrating planning, tracking, and control under realistic dynamics.

**Mechanism:**

* Uses a fixed-step deterministic integration loop for $T / \Delta t$ iterations.
* Sequentially invokes the MPC tracker, applies controls, updates vehicle states, and logs diagnostics.
* Terminates when vehicle velocity drops below a configurable threshold after reaching the end of the reference path.

**Output:** A structured `SimulationLog` containing per-step state vectors, control commands, MPC predictions, and tracking errors.

**Scientific value:** Enables repeatable experiments, convergence analysis, and consistent quantitative evaluation.

---

### 2.8. Visualization (`src/vis/`)

**Purpose:** Render simulation and planning outputs into interpretable figures and animations.

**Components:**

* **Static dashboards:** Combine map views, velocity and acceleration plots, steering, error metrics, and force diagrams.
* **Animated trajectories:** Display vehicle evolution, MPC horizon predictions, and obstacle interaction in real time.
* **FMT debug visualizations:** Illustrate sampling and tree expansion behavior for algorithmic inspection.

**Scientific function:** Provides reproducible visual evidence of planner and controller performance for publication or analysis.

---

### 2.9. Logging (`src/common/logging_utils.py`)

**Purpose:** Centralize logging configuration for all modules.

**Features:**

* Uniform timestamping, module name tagging, and severity level formatting.
* Dual-channel logging: console output and synchronized `.log` files.
* Ensures perfect temporal alignment between runtime messages and stored simulation artifacts.

**Result:** Guarantees traceability and reproducibility across multiple experimental runs.

---

## 3. Testing and Validation

**Location:** `tests/`

**Coverage:**

* **Unit tests:** Validate individual module correctness (mapping, dynamics, FMT connectivity, MPC feasibility).
* **Integration tests:** Verify complete end-to-end functionality from configuration loading to visualization artifact generation.
* Includes both **conservative** (slow-speed, tight-clearance) and **high-speed** scenarios to test numerical stability and control robustness.

**Scientific guarantee:** Ensures every subsystem adheres to defined physical, mathematical, and numerical contracts under varying operating conditions.

---

## 4. Summary

The **System Architecture** is engineered for modularity, determinism, and scientific reproducibility.
Key architectural principles:

1. **Isolated scientific domains:** Each package encapsulates a single research concern.
2. **Deterministic execution:** Fixed-seed randomization and discretized integration grids eliminate non-determinism.
3. **Explicit configuration contracts:** YAML-driven dataclasses ensure transparency between configuration and execution.
4. **Layered design:** Separation of mapping, planning, control, and visualization enables targeted experimentation.
5. **Comprehensive testing:** Unit and integration coverage verify algorithmic correctness across conditions.

This structured, modular framework supports research reproducibility, facilitates algorithmic substitution, and provides a robust foundation for autonomous system development and evaluation.
