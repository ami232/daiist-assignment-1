# Assignment 1 Report


- **Name**: Yan Guy Patrick Bernard
- **Student ID**: 20202
- **Email**: ybernard.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

This dataset is a Synthetic missile trajectory simulation dataset that I found on Kaggle. It has three parquet files :
- full_trajectories
- radar_marks_clean
- radar_marks_noisy

basically they aim to simulate the trajectory of two type of rockets :
- Ballistic projectile: an unpowered target with a high initial velocity.
- Rocket powered target: a powered target with a lower initial velocity and a short active thrust phase.

Where this gets interesting is that we have both the full trajectory data per launch but also radar pings which are a subsample of each trajectory at different points in time (7 pings).

Meaning, the data that our radar has managed to collect over the entire trajectory (simulates real life situation)

These are our variables
| Column | Description | Unit |
|---|---|---|
| `t` | Time from the beginning of the simulated trajectory | seconds |
| `x` | Cartesian X coordinate | meters |
| `y` | Cartesian Y coordinate | meters |
| `z` | Altitude / Cartesian Z coordinate | meters |
| `vx` | Velocity component along X | m/s |
| `vy` | Velocity component along Y | m/s |
| `vz` | Velocity component along Z | m/s |
| `v` | Total speed magnitude | m/s |
| `trajectory_id` | Unique trajectory identifier | - |
| `is_rocket` | Target class flag: `True` for rocket powered targets, `False` for ballistic projectiles | - |


## Business / real-life framing

With war waging around the world more and more as time goes by, weapons such as drones and missiles are at the center of conflicts as they aim to destroy or enable military, industrial or even civillian targets. The idea with this project is to build the first step in detecting threats in advance. With the idea of later doing the necessary work to intercept such threats.

**Scenario:** A ground based air defence radar detects an incoming object. It obtains exactly **7 position/velocity pings over a ~36 second window**, then loses track. The system must immediately answer two questions:

1. **Where will it land?** `(x_end, y_end)` in metres — to position interceptors or issue evacuation orders.
2. **How much time remains?** `t_remaining` in seconds — to determine intercept feasibility.


## Data preparation & feature engineering
Starting point: 7 radar pings × 7 variables (x, y, z, vx, vy, vz, v) = 49 raw features per trajectory (pivoted wide)
in order to get one row per trajectory.

Then I looked at making new features based on actual physics not just interactions or other ML technics, I also did this first and ran the models which ended up having great metrics.

| Feature | Formula | Physical rationale |
|---|---|---|
| `h_speed_last` | `√(vx_p6² + vy_p6²)` | Horizontal speed at last ping directly determines how far the object can travel before impact |
| `h_range_obs` | `√((x_p6−x_p0)² + (y_p6−y_p0)²)` | Distance covered during radar window proxy for total range |
| `est_apogee_km` | `z_p6 + vz_p6²/(2g)` if ascending, else `z_p6` | Kinematic estimate of max altitude; ballistics peak early, rockets later |
| `v_slope` | Linear fit slope of `v` over 7 pings | Rockets accelerate (positive slope), ballistics decelerate (negative) |
| `bearing_rad` | `arctan2(vy_p6, vx_p6)` | Direction of travel combined with last position, determines landing quadrant |

### Preprocessing
- `StandardScaler` fitted on training set only, applied to test set (no leakage).
- Targets (`x_end`, `y_end`, `t_remaining`) were also scaled for PyTorch training, then inverse transformed for evaluation.

## Split Strategy

```
train_test_split(X, y, test_size=0.2, random_state=42)
→ Train: 3,999 trajectories   Test: 1,000 trajectories
```

**Important:** the raw radar file has 7 rows per trajectory (one per ping). The 7 pings must be consumed in order — ping 0 through ping 6 — and must never be split across train and test (data leakage). The pipeline handles this by pivoting all 7 pings into a single row per trajectory. 

Essentially we transformed 7 rows per trajectory to one big row with the features of all 7 pings as columns.

By the time `train_test_split` is called, the unit is one complete trajectory. The sequential order of pings is preserved in the feature names (`x_p0 → x_p6`, `vz_p0 → vz_p6`, etc.). A trajectory's pings are never separated across the split.

**Why random (not time based) split:** the trajectories are independent parallel simulations no trajectory's outcome depends on another's, and there is no meaningful time ordering between trajectory IDs.

## Modeling: three implementations, one model

All three methods implement the same model: **multi-output Ridge regression** (L2 regularisation, α=1.0) predicting `(x_end, y_end, t_remaining)`.

### Metrics on test set (n=1,000)

| Model | RMSE 2D (m) | RMSE Δt (s) | R² x | R² y | R² Δt |
|---|---|---|---|---|---|
| **sklearn Ridge** | 4,102 | 2.4 s | 0.9862 | 0.9852 | 0.9843 |
| **PyTorch Scratch (SGD)** | 4,582 | 2.6 s | 0.9828 | 0.9815 | 0.9830 |
| **PyTorch nn.Module (Adam)** | 4,141 | 2.4 s | 0.9860 | 0.9848 | 0.9844 |

All three implement the same linear model, the gap is **optimisation, not model capacity**.

- **sklearn Ridge** solves the normal equations analytically: `W* = (XᵀX + αI)⁻¹ Xᵀy`. This is the exact global minimum and the reference solution.
- **nn.Module + Adam** uses adaptive per parameter learning rates. It converges faster than plain SGD but does not reach the exact OLS solution after a fixed number of epochs (3,000 here). Gap: ~40 m RMSE 2D vs sklearn. With more epochs its would end up at the same result.
- **From-scratch SGD** uses a fixed learning rate (0.05) with no momentum or adaptation. It converges most slowly and shows the largest gap (~480 m). The loss curve in Tab 4 shows it is still decreasing at epoch 3,000.
- **Given unlimited epochs**, all three converge to the same weights.

## Limitations & next steps

1. **Synthetic data:** The dataset was generated, not collected from a real radar system. Physical parameters (drag, thrust profile, atmospheric conditions) are simplified. The high R (0.986) partly reflects this regularity, a real radar dataset would be messier. We could make the task harder by generating topographical data such as altitude of rocket launch or altitude of area of landing, different rocket types with largely different characteristics and more.

2. **Linear model ceiling:** With R ≈ 0.986, most of the remaining error (RMSE ≈ 4 km) likely comes from rockets in their powered phase, where non linear thrust effects make the trajectory unpredictable from a linear model
## Generative AI use disclosure

I used claude code and I used gemini.

Claude code :
- Gradio app, I am not a big frontend guru and I just wanted to make the dashboard look cool given the theme, so the graphs etc... are claude generated code
- I also used claude code to automatically write the models, metrics, train curve data into their respective folder. 

Gemini :
- I used gemini to pull the feature engineering formulas since i didn't want to say nonsense and disrupt the model with illogical variables.
