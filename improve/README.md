# FL-QECO (improve/)

Federated extension of QECO under **the same number of edge nodes as the root experiment** (`FLConfig.N_EDGE` is read from root [`Config.N_EDGE`](../Config.py)). **Multiple MDs**; periodic **FedAvg** on the **shared backbone** (TensorFlow scope: `eval_net` → `l0` LSTM, `l1`, `l12`). Each MD keeps a **private replay buffer** and **personalized Dueling heads** (`Value` / `Advantage`). **FedProx** is implemented as a proximal term \(\frac{\mu}{2}\|w_{\text{backbone}}-w^{(t)}\|^2\) w.r.t. a frozen anchor updated only at aggregation rounds.

## Why `improve/` exists

- Root [`D3QN.py`](../D3QN.py) calls `tf.reset_default_graph()` inside `_build_net()`, which is unsafe when constructing **one network per MD**. Here each agent uses its own **`tf.Graph()` + `tf.Session(graph=...)`**.
- All FL experiment scripts and logs live under **`improve/`** only.

## Run

From repository root (Windows / Linux):

```bash
python improve/main_fl_qeco.py --mode fed
python improve/main_fl_qeco.py --mode local
```

- **`fed`**: FedAvg on backbone + FedProx (`FEDPROX_MU` in `config_fl.py`) + periodic aggregation (`AGG_INTERVAL_RL_STEPS`, `PARTICIPATION_RATIO`).
- **`local`**: same code path, **no aggregation**, **μ = 0** (independent per-MD QECO-style training with fixed multi-graph setup).

Optional overrides:

```bash
python improve/main_fl_qeco.py --mode fed --agg-interval 50 --participation 0.5 --fedprox-mu 0.02 --episodes 200
```

Outputs:

- `improve/results/metrics_<mode>_...csv` — per-episode QoE, delay, energy, drops, plus FL metadata when `fed`.
- `improve/results/Performance_Chart_<mode>.png` — smoothed curves (if matplotlib available).

## Cold-start experiment

After a short joint training, compares **reset head + empty buffer + inject global backbone** vs **reset without backbone injection**:

```bash
python improve/experiment_cold_start.py --episodes-warm 80 --episodes-cool 40
```

## Sensitivity analysis

Sweep `AGG_INTERVAL_RL_STEPS`, `PARTICIPATION_RATIO`, and `FEDPROX_MU` via CLI flags on `main_fl_qeco.py` or by editing [`improve/config_fl.py`](config_fl.py). CSV columns include `agg_interval`, `participation_ratio`, `fedprox_mu`, and approximate `comm_bytes_round`.

## Dependencies

Same as project root: TensorFlow 1.x (`tensorflow.compat.v1`), NumPy, Matplotlib. [`MEC_Env.py`](../MEC_Env.py) is imported from the parent package path at runtime.
