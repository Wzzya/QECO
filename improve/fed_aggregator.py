"""
EN-side FedAvg on shared D3QN backbone (list of numpy tensors per MD).
"""
import numpy as np


def backbone_num_floats(weights_list):
    return int(sum(w.size for w in weights_list))


def fedavg_backbone(weight_lists):
    """
    weight_lists: list of K entries; each entry is list of arrays from one MD (same length/shapes).
    Returns list of averaged arrays.
    """
    if not weight_lists:
        raise ValueError("empty weight_lists")
    n = len(weight_lists)
    nvars = len(weight_lists[0])
    out = []
    for j in range(nvars):
        acc = np.zeros_like(weight_lists[0][j], dtype=np.float64)
        for k in range(n):
            acc += weight_lists[k][j].astype(np.float64)
        out.append((acc / n).astype(np.float32))
    return out


def sample_participants(n_ue, participation_ratio, rng):
    """At least one MD participates."""
    k = max(1, int(np.ceil(n_ue * float(participation_ratio))))
    k = min(k, n_ue)
    return sorted(rng.choice(n_ue, size=k, replace=False).tolist())


class EdgeFedAggregator:
    """Stateful helper for logging; aggregation is pure functions above."""

    def __init__(self, rng=None):
        self.rng = np.random.RandomState() if rng is None else rng

    def aggregate_and_broadcast(self, ue_rl_list, participant_indices):
        """
        FedAvg backbone from participants, then write global backbone + anchors
        to every MD and sync target backbone from eval (heads unchanged).
        """
        weights = [ue_rl_list[i].get_backbone_weights() for i in participant_indices]
        w_bar = fedavg_backbone(weights)
        for agent in ue_rl_list:
            agent.set_backbone_weights(w_bar)
            agent.set_anchor_from_backbone_list(w_bar)
            agent.sync_target_backbone_from_eval()
        return w_bar

    def comm_bytes_estimate(self, n_floats, n_participants, n_devices):
        """Rough upper bound: each participant uploads + all devices download."""
        bytes_per = n_floats * 4
        upload = n_participants * bytes_per
        download = n_devices * bytes_per
        return int(upload + download)
