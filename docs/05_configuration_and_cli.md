# Configuration and CLI Workflows

Scenario YAML files configure the entire pipeline. Each scenario contains five core sections whose parameters are
consumed by strongly typed dataclasses in `src/common/config.py`.

1. **map** – points to a PNG (white = driveable, black = occupied), specifies resolution (`meters_per_cell`), and defines
the world origin of the bitmap.
2. **start / goal** – three-element poses `[x, y, yaw]` expressed in metres and radians. For the bundled `map_raw.png`
   (80 m × 40 m) the conservative scenario starts at `(5, 5)` and targets `(35, 32)` while the high-speed scenario
   finishes near `(72, 32)`.
3. **planner_config** – sampling radius, neighbour count, smoothing options, and safety margins for FMT.
4. **tracker_config** – MPC horizon length, timestep, weighting matrices, and cruise-speed targets.
5. **visualization** – toggles for planner debug windows, animation outputs, file locations, and vehicle drawing params.

```yaml
map:
  png_path: maps/map_raw.png
  meters_per_cell: 0.2
  origin: [0.0, 0.0]
start: [5.0, 5.0, 0.0]
goal: [35.0, 32.0, 0.0]
visualization:
  show_planner_debug: true
  animate_planner: true
  animate_simulation: true
  vehicle_draw:
    wheelbase: 2.7
    width: 1.55
    length: 4.5
    front_overhang: 0.9
    rear_overhang: 0.8
    wheel_track: 1.2
    tire_radius: 0.33
    tire_width: 0.22
```

The `meters_per_cell` value $m_{\mathrm{cell}}$ acts as a scale factor when converting between pixel indices $(i, j)$ and
metric coordinates $(x, y)$:

$$
  x = x_0 + i \cdot m_{\mathrm{cell}}, \qquad y = y_0 + j \cdot m_{\mathrm{cell}},
$$

where $(x_0, y_0)$ denotes the `origin` specified in the YAML file. This mapping is used symmetrically when rendering
simulation results back onto the occupancy grid.

## CLI Modes

The entry point `python -m src.cli <config> <mode>` (or the installed console script `fmt-cli <config> <mode>`) supports
four execution modes:

* `plan_only` – run mapping + FMT to inspect the planned corridor and optional debug plot.
* `track_only` – run planning and generate the first MPC command without stepping the simulator.
* `run_e2e` – complete pipeline from planning through simulation and visualisation.
* `export_map` – inflate and export the occupancy grid referenced by the scenario.

Enable or disable plots and animations in the `visualization` block without modifying code. Output directories default to
`artifacts/<scenario_name>/` and are created automatically.
