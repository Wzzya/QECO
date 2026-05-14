"""
Cold-start comparison: after warm FL training, reset MD0 (memory + heads) and
either inject FedAvg backbone (FL-assisted) or keep random backbone (baseline).
All code stays under improve/.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MEC_Env import MEC  # noqa: E402

from improve.config_fl import FLConfig  # noqa: E402
from improve.d3qn_fl import DuelingDoubleDeepQNetworkFL  # noqa: E402
from improve.fed_aggregator import fedavg_backbone  # noqa: E402
from improve.main_fl_qeco import train_fl  # noqa: E402


def build_agents(env, fedprox_mu):
    ue_RL_list = []
    for _ in range(FLConfig.N_UE):
        ue_RL_list.append(
            DuelingDoubleDeepQNetworkFL(
                env.n_actions,
                env.n_features,
                env.n_lstm_state,
                env.n_time,
                learning_rate=FLConfig.LEARNING_RATE,
                reward_decay=FLConfig.REWARD_DECAY,
                e_greedy=FLConfig.E_GREEDY,
                replace_target_iter=FLConfig.N_NETWORK_UPDATE,
                memory_size=FLConfig.MEMORY_SIZE,
                batch_size=FLConfig.BATCH_SIZE,
                fedprox_mu=fedprox_mu,
            )
        )
    return ue_RL_list


def snapshot_global_backbone(ue_RL_list):
    weights = [a.get_backbone_weights() for a in ue_RL_list]
    return fedavg_backbone(weights)


def run_cool_phase(env, ue_RL_list, num_episode, inject_backbone, w_global, rng, tag, out_dir):
    """Reset MD0 memory/heads; optionally load w_global into MD0 backbone+anchor+sync target."""
    ue_RL_list[0].reset_personalized_head_and_memory(reinit_heads=True)
    if inject_backbone and w_global is not None:
        ue_RL_list[0].set_backbone_weights([w.copy() for w in w_global])
        ue_RL_list[0].set_anchor_from_backbone_list([w.copy() for w in w_global])
        ue_RL_list[0].sync_target_backbone_from_eval()
    else:
        ue_RL_list[0].reinitialize_eval_backbone_random()

    csv_path = os.path.join(out_dir, "coldstart_%s.csv" % tag)
    chart_path = os.path.join(out_dir, "coldstart_%s.png" % tag)
    return train_fl(
        ue_RL_list,
        env,
        num_episode,
        fed_mode=True,
        fedprox_mu=FLConfig.FEDPROX_MU,
        agg_interval=FLConfig.AGG_INTERVAL_RL_STEPS,
        participation_ratio=FLConfig.PARTICIPATION_RATIO,
        csv_path=csv_path,
        chart_path=chart_path,
        rng=rng,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes-warm", type=int, default=80)
    parser.add_argument("--episodes-cool", type=int, default=40)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    os.makedirs(FLConfig.RESULTS_DIR, exist_ok=True)

    env = MEC(
        FLConfig.N_UE,
        FLConfig.N_EDGE,
        FLConfig.N_TIME,
        FLConfig.N_COMPONENT,
        FLConfig.MAX_DELAY,
    )

    # --- Warm federated training ---
    ue_list = build_agents(env, FLConfig.FEDPROX_MU)
    warm_csv = os.path.join(FLConfig.RESULTS_DIR, "coldstart_warm.csv")
    warm_chart = os.path.join(FLConfig.RESULTS_DIR, "coldstart_warm.png")
    train_fl(
        ue_list,
        env,
        args.episodes_warm,
        fed_mode=True,
        fedprox_mu=FLConfig.FEDPROX_MU,
        agg_interval=FLConfig.AGG_INTERVAL_RL_STEPS,
        participation_ratio=FLConfig.PARTICIPATION_RATIO,
        csv_path=warm_csv,
        chart_path=warm_chart,
        rng=rng,
    )
    w_global = snapshot_global_backbone(ue_list)
    print("Warm done. Global backbone tensors:", len(w_global))

    # --- Cool A: inject backbone to MD0 after reset ---
    env_a = MEC(
        FLConfig.N_UE,
        FLConfig.N_EDGE,
        FLConfig.N_TIME,
        FLConfig.N_COMPONENT,
        FLConfig.MAX_DELAY,
    )
    agents_a = build_agents(env_a, FLConfig.FEDPROX_MU)
    # Copy warm-trained weights to agents_a (same architecture) for all MDs except we will reset MD0 only
    for i, (wa, wb) in enumerate(zip(agents_a, ue_list)):
        wa.set_backbone_weights(wb.get_backbone_weights())
        wa.set_anchor_from_backbone_list(wb.get_backbone_weights())
        wa.sync_target_backbone_from_eval()
        # copy epsilon / learn counter optional - skip for harsher cold start
    qoe_inject = np.mean(
        run_cool_phase(
            env_a,
            agents_a,
            args.episodes_cool,
            inject_backbone=True,
            w_global=w_global,
            rng=np.random.RandomState(args.seed + 1),
            tag="inject_md0",
            out_dir=FLConfig.RESULTS_DIR,
        )
    )

    # --- Cool B: no injection (random backbone after reset) ---
    env_b = MEC(
        FLConfig.N_UE,
        FLConfig.N_EDGE,
        FLConfig.N_TIME,
        FLConfig.N_COMPONENT,
        FLConfig.MAX_DELAY,
    )
    agents_b = build_agents(env_b, FLConfig.FEDPROX_MU)
    for i, (wa, wb) in enumerate(zip(agents_b, ue_list)):
        wa.set_backbone_weights(wb.get_backbone_weights())
        wa.set_anchor_from_backbone_list(wb.get_backbone_weights())
        wa.sync_target_backbone_from_eval()
    qoe_no_inject = np.mean(
        run_cool_phase(
            env_b,
            agents_b,
            args.episodes_cool,
            inject_backbone=False,
            w_global=None,
            rng=np.random.RandomState(args.seed + 2),
            tag="no_inject_md0",
            out_dir=FLConfig.RESULTS_DIR,
        )
    )

    summary_path = os.path.join(FLConfig.RESULTS_DIR, "coldstart_summary.txt")
    with open(summary_path, "w") as f:
        f.write("mean QoE cool phase (MD0 reset; others continue)\n")
        f.write("inject_global_backbone: %.4f\n" % qoe_inject)
        f.write("no_inject (random backbone after reset): %.4f\n" % qoe_no_inject)
    print("Wrote", summary_path)
    print("Mean QoE inject:", qoe_inject, "no inject:", qoe_no_inject)


if __name__ == "__main__":
    main()
