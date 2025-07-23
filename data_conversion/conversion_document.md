# Data Conversion Pipeline: Scenario to Time-Series Descriptors

This document outlines the process for converting waymo motion data from a tfrecord into a time-series of structured descriptors for driving with llmls.

---

## What is a "Sample"?

Given one sid and one ego agent id, we have a unique driving scenario that is considered as one sample. 

One sample has 19 timesteps so it will be converted to a list of 19 dicts of descriptor format data.

A **single sample** has the combination of:
1.  A static **map** of the environment (`Map Data`).
2.  The **ego vehicle's** full trajectory (`Ego Trajectory`).
3.  The trajectories of all **nearby agents** (`Nearby Agent Trajectories`).


---

## Conversion Process

The goal is to transform one such sample into a **list of 19 data snapshots**. Each snapshot is a dictionary representing the state of the entire scene at a specific time step.

### Step 1: Trajectory Preprocessing

Before any calculations, every agent's trajectory (a `(19, 2)` numpy array) is cleaned to handle invalid `[-1, -1]` points. This is done by the `_interpolate_invalid_points` function from convert_data_utils.py.

-   **Interpolation**: Gaps of invalid points *between* two valid points are filled with evenly spaced points on a straight line.
-   **Extrapolation**: Gaps at the beginning or end of a trajectory are filled by projecting the agent's path forward or backward based on its last known velocity.

This ensures every agent has a valid `(x, y)` coordinate for all 19 time steps.

### Step 2: Time-Series Generation

The pipeline iterates from time `t=0` to `t=18`. In each iteration, it calculates the instantaneous state for every agent and generates one complete data snapshot. This is done by `convert_to_descriptor_format` function from convert_data_utils

-   **Initial State (t=0)**: All dynamic properties (speed, acceleration, heading, dx, dy) are initialized to **0.0**.
-   **Subsequent States (t>0)**: Dynamics are calculated based on the change in position from previous time steps.

### Step 3: Final Descriptor Formatting

For each of the 19 time steps, the following four `numpy` arrays are created and bundled into a dictionary.

| Descriptor               | Shape      | Description                                                                                                                                                                                                                                                          |
| :----------------------- | :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ego_vehicle_descriptor` | `(7,)`     | The full state of the ego vehicle: `[ACCEL, SPEED, X, Y, HEADING, DX, DY]`.                                                                                                                                                                                           |
| `vehicle_descriptors`    | `(30, 6)`  | The state of up to 30 nearby vehicles, padded with zeros. Each row is `[SPEED, X, Y, DX, DY, HEADING]`.                                                                                                                                                               |
| `pedestrian_descriptors` | `(20, 6)`  | The state of up to 20 nearby pedestrians, padded with zeros.                                                                                                                                                                                                         |
| `road_descriptors`       | `(60, 46)` | Describes up to 60 road segments relative to the ego's *current* position. Each row contains **10 points** (`X1,Y1...`), **10 invalid flags**, and a **one-hot encoded type vector**. If a segment has >10 points, a window is selected based on proximity to the ego. |

The final output for one input sample is a list containing 19 of these structured dictionaries, representing the scene in full chronological order.



# How to use the code?

`convert_data` function from convert_data.py takes in one tfrecord file path and iterate through all the samples. For each sample, a list of 19 dicts data in time sequence will be print.

For example, `/p/liverobotics/waymo_open_dataset_motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150` has 49 scenarios (samples), so it will provide 49 lists. Each lists has 19 dicts so 49*19 datapoints in total.
