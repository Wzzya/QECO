"""
FL-QECO experiment configuration (multi-MD, FedProx on shared backbone).
Edge server count matches root Config.py (same as main.py experiment).
Does not modify root Config.py; FL-specific settings live here.
"""
import os
import sys

# Repo root: allow "from Config import Config" when improve is imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from Config import Config as _RootConfig  # noqa: E402


class FLConfig(object):
    # --- Base scenario (EN count aligned with root main.py / Config.py) ---
    N_UE = 20
    N_EDGE = _RootConfig.N_EDGE
    N_EPISODE = 1000
    N_TIME_SLOT = 100
    MAX_DELAY = 10
    N_TIME = N_TIME_SLOT + MAX_DELAY  # must match MEC_Env time horizon
    DURATION = 0.1
    TASK_ARRIVE_PROB = 0.3

    UE_COMP_CAP = 2.6
    UE_TRAN_CAP = 14
    EDGE_COMP_CAP = 42
    UE_ENERGY_STATE = [0.25, 0.50, 0.75]
    UE_COMP_ENERGY = 2
    UE_TRAN_ENERGY = 2.3
    UE_IDLE_ENERGY = 0.1
    EDGE_COMP_ENERGY = 5
    TASK_COMP_DENS = [0.197, 0.297, 0.397]
    TASK_MIN_SIZE = 1
    TASK_MAX_SIZE = 7
    N_COMPONENT = 1

    # --- DRL (match root Config defaults unless overridden) ---
    LEARNING_RATE = 0.01
    REWARD_DECAY = 0.9
    E_GREEDY = 0.99
    N_NETWORK_UPDATE = 200
    MEMORY_SIZE = 500
    BATCH_SIZE = 32
    LEARN_EVERY_RL_STEPS = 10
    LEARN_START_RL_STEP = 200

    # --- Federated learning (EN-side logic in trainer) ---
    # Aggregate every AGG_INTERVAL_RL_STEPS global RL steps (after learn block).
    AGG_INTERVAL_RL_STEPS = 100
    PARTICIPATION_RATIO = 1.0  # fraction of MDs uploading each round (min 1 device)
    FEDPROX_MU = 0.01  # proximal weight; set 0 in local-only mode
    # Bytes estimate: 2 * n_participants * backbone_bytes (upload + download approx)
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
