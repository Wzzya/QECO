"""
FL-QECO training entry: single EN + multi-MD, FedAvg backbone + FedProx (optional),
independent graphs per MD. Results under improve/results/.
"""
from __future__ import print_function

import argparse
import csv
import os
import sys

import numpy as np

# Repo root on path for MEC_Env / Config (used inside MEC)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from MEC_Env import MEC  # noqa: E402

from improve.config_fl import FLConfig  # noqa: E402
from improve.d3qn_fl import DuelingDoubleDeepQNetworkFL  # noqa: E402
from improve.fed_aggregator import (  # noqa: E402
    EdgeFedAggregator,
    backbone_num_floats,
    sample_participants,
)


def normalize(parameter, minimum, maximum):
    return (parameter - minimum) / (maximum - minimum)


def QoE_Function(
    delay,
    max_delay,
    unfinish_task,
    ue_energy_state,
    ue_comp_energy,
    ue_trans_energy,
    edge_comp_energy,
    ue_idle_energy,
):
    edge_energy = next((e for e in edge_comp_energy if e != 0), 0)
    _ = edge_energy  # kept for parity with main.py
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    _ = idle_energy
    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))
    reward = max_delay * 4
    if unfinish_task:
        qoe = -cost
    else:
        qoe = reward - cost
    return qoe


def Drop_Count(ue_RL_list, episode, env):
    drrop = 0
    for time_index in range(100):
        drrop = drrop + sum(env.unfinish_task[time_index])
    drrop_delay10 = 0
    for i in range(len(ue_RL_list)):
        for j in range(len(ue_RL_list[i].delay_store[episode])):
            if ue_RL_list[i].delay_store[episode][j] == 10:
                drrop_delay10 = drrop_delay10 + 1
    return drrop


def Cal_QoE(ue_RL_list, episode):
    episode_sum_reward = sum(sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list)
    return episode_sum_reward / len(ue_RL_list)


def Cal_Delay(ue_RL_list, episode):
    avg_delay_in_episode = []
    for i in range(len(ue_RL_list)):
        for j in range(len(ue_RL_list[i].delay_store[episode])):
            if ue_RL_list[i].delay_store[episode][j] != 0:
                avg_delay_in_episode.append(ue_RL_list[i].delay_store[episode][j])
    if not avg_delay_in_episode:
        return 0.0
    return sum(avg_delay_in_episode) / len(avg_delay_in_episode)


def Cal_Energy(ue_RL_list, episode):
    energy_ue_list = [sum(ue_RL.energy_store[episode]) for ue_RL in ue_RL_list]
    return sum(energy_ue_list) / len(energy_ue_list)


def _moving_average(y, window=31):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy()
    w = max(3, min(int(window), n))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(y, kernel, mode="same")


def _style_paper_axis(ax):
    ax.grid(True, linestyle="--", color="0.3", linewidth=0.85, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors="black")


def train_fl(
    ue_RL_list,
    env,
    num_episode,
    fed_mode,
    fedprox_mu,
    agg_interval,
    participation_ratio,
    csv_path,
    chart_path,
    rng,
):
    aggregator = EdgeFedAggregator(rng=rng)
    n_floats = backbone_num_floats(ue_RL_list[0].get_backbone_weights())

    avg_QoE_list = []
    avg_delay_list = []
    energy_cons_list = []
    num_drop_list = []

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(
            [
                "episode",
                "avg_qoe",
                "avg_delay",
                "avg_energy",
                "num_drop",
                "mode",
                "fedprox_mu",
                "agg_interval",
                "participation_ratio",
                "comm_bytes_round",
            ]
        )

    RL_step = 0
    last_comm_bytes = 0

    for episode in range(num_episode):
        print("\nEpisode:", episode, "Epsilon:", ue_RL_list[0].epsilon)

        bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob
        bitarrive_size = bitarrive_size * (np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob)
        bitarrive_size[-env.max_delay :, :] = np.zeros([env.max_delay, env.n_ue])

        bitarrive_dens = np.zeros([env.n_time, env.n_ue])
        for i in range(len(bitarrive_size)):
            for j in range(len(bitarrive_size[i])):
                if bitarrive_size[i][j] != 0:
                    bitarrive_dens[i][j] = FLConfig.TASK_COMP_DENS[
                        np.random.randint(0, len(FLConfig.TASK_COMP_DENS))
                    ]

        history = []
        for time_index in range(env.n_time):
            history.append([])
            for ue_index in range(env.n_ue):
                history[time_index].append(
                    {
                        "observation": np.zeros(env.n_features),
                        "lstm": np.zeros(env.n_lstm_state),
                        "action": np.nan,
                        "observation_": np.zeros(env.n_features),
                        "lstm_": np.zeros(env.n_lstm_state),
                    }
                )
        reward_indicator = np.zeros([env.n_time, env.n_ue])

        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens)

        while True:
            action_all = np.zeros([env.n_ue])
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if np.sum(observation) == 0:
                    action_all[ue_index] = 0
                else:
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    if observation[0] != 0:
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            observation_all_, lstm_state_all_, done = env.step(action_all)

            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index, :])

            process_delay = env.process_delay
            unfinish_task = env.unfinish_task

            for ue_index in range(env.n_ue):
                history[env.time_count - 1][ue_index]["observation"] = observation_all[ue_index, :]
                history[env.time_count - 1][ue_index]["lstm"] = np.squeeze(lstm_state_all[ue_index, :])
                history[env.time_count - 1][ue_index]["action"] = action_all[ue_index]
                history[env.time_count - 1][ue_index]["observation_"] = observation_all_[ue_index]
                history[env.time_count - 1][ue_index]["lstm_"] = np.squeeze(lstm_state_all_[ue_index, :])

                update_index = np.where((1 - reward_indicator[:, ue_index]) * process_delay[:, ue_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        ue_RL_list[ue_index].store_transition(
                            history[time_index][ue_index]["observation"],
                            history[time_index][ue_index]["lstm"],
                            history[time_index][ue_index]["action"],
                            QoE_Function(
                                process_delay[time_index, ue_index],
                                env.max_delay,
                                unfinish_task[time_index, ue_index],
                                env.ue_energy_state[ue_index],
                                env.ue_comp_energy[time_index, ue_index],
                                env.ue_tran_energy[time_index, ue_index],
                                env.edge_comp_energy[time_index, ue_index],
                                env.ue_idle_energy[time_index, ue_index],
                            ),
                            history[time_index][ue_index]["observation_"],
                            history[time_index][ue_index]["lstm_"],
                        )
                        r = QoE_Function(
                            process_delay[time_index, ue_index],
                            env.max_delay,
                            unfinish_task[time_index, ue_index],
                            env.ue_energy_state[ue_index],
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy[time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index],
                        )
                        ue_RL_list[ue_index].do_store_reward(episode, time_index, r)
                        ue_RL_list[ue_index].do_store_delay(episode, time_index, process_delay[time_index, ue_index])
                        ue_RL_list[ue_index].do_store_energy(
                            episode,
                            time_index,
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy[time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index],
                        )
                        reward_indicator[time_index, ue_index] = 1

            RL_step += 1
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            if (RL_step > FLConfig.LEARN_START_RL_STEP) and (RL_step % FLConfig.LEARN_EVERY_RL_STEPS == 0):
                mu = fedprox_mu if fed_mode else 0.0
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn(fedprox_mu=mu)
                if fed_mode and (RL_step % agg_interval == 0):
                    participants = sample_participants(env.n_ue, participation_ratio, rng)
                    aggregator.aggregate_and_broadcast(ue_RL_list, participants)
                    last_comm_bytes = aggregator.comm_bytes_estimate(
                        n_floats, len(participants), env.n_ue
                    )
                    print(
                        "FL round @ RL_step",
                        RL_step,
                        "participants",
                        len(participants),
                        "comm_est_bytes",
                        last_comm_bytes,
                    )

            if done:
                avg_delay = Cal_Delay(ue_RL_list, episode)
                avg_energy = Cal_Energy(ue_RL_list, episode)
                avg_QoE = Cal_QoE(ue_RL_list, episode)
                num_drop = env.drop_trans_count + env.drop_edge_count + env.drop_ue_count

                avg_QoE_list.append(avg_QoE)
                avg_delay_list.append(avg_delay)
                energy_cons_list.append(avg_energy)
                num_drop_list.append(num_drop)

                with open(csv_path, "a", newline="") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(
                        [
                            episode,
                            avg_QoE,
                            avg_delay,
                            avg_energy,
                            num_drop,
                            "fed" if fed_mode else "local",
                            fedprox_mu if fed_mode else 0.0,
                            agg_interval if fed_mode else 0,
                            participation_ratio if fed_mode else 0.0,
                            last_comm_bytes if fed_mode else 0,
                        ]
                    )

                if episode % 10 == 0 and len(avg_QoE_list) >= 2:
                    plt.rcParams.update(
                        {
                            "font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
                            "axes.edgecolor": "black",
                            "axes.linewidth": 1.0,
                        }
                    )
                    ma_window = max(7, min(51, len(avg_QoE_list) // 15 or 7))
                    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
                    fig.suptitle("FL-QECO / QoE-D3QN (improve)", fontsize=14, y=0.92)
                    x_q = np.arange(len(avg_QoE_list))
                    axs[0].plot(x_q, _moving_average(avg_QoE_list, ma_window), color="#1f4e79", label="Avg QoE")
                    axs[0].set_ylabel("Average QoE")
                    _style_paper_axis(axs[0])
                    axs[0].legend(loc="lower right")
                    axs[1].plot(
                        np.arange(len(avg_delay_list)),
                        _moving_average(avg_delay_list, ma_window),
                        color="#2e7d32",
                    )
                    axs[1].set_ylabel("Avg Delay")
                    _style_paper_axis(axs[1])
                    axs[2].plot(
                        np.arange(len(energy_cons_list)),
                        _moving_average(energy_cons_list, ma_window),
                        color="#c00000",
                    )
                    axs[2].set_ylabel("Energy")
                    _style_paper_axis(axs[2])
                    axs[3].plot(
                        np.arange(len(num_drop_list)),
                        _moving_average(num_drop_list, ma_window),
                        color="#7030a0",
                    )
                    axs[3].set_ylabel("Drops")
                    _style_paper_axis(axs[3])
                    plt.tight_layout()
                    plt.savefig(chart_path, dpi=120)
                    plt.close(fig)

                print(
                    "Episode done | drops:",
                    num_drop,
                    "avg_QoE",
                    "%.2f" % avg_QoE,
                    "avg_delay",
                    "%.2f" % avg_delay,
                )
                break

    return avg_QoE_list


def main():
    parser = argparse.ArgumentParser(description="FL-QECO (improve/)")
    parser.add_argument("--mode", choices=["fed", "local"], default="fed")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--agg-interval", type=int, default=None)
    parser.add_argument("--participation", type=float, default=None)
    parser.add_argument("--fedprox-mu", type=float, default=None)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    fed_mode = args.mode == "fed"
    num_episode = args.episodes if args.episodes is not None else FLConfig.N_EPISODE
    agg_interval = args.agg_interval if args.agg_interval is not None else FLConfig.AGG_INTERVAL_RL_STEPS
    participation_ratio = (
        args.participation if args.participation is not None else FLConfig.PARTICIPATION_RATIO
    )
    fedprox_mu = args.fedprox_mu if args.fedprox_mu is not None else FLConfig.FEDPROX_MU
    if not fed_mode:
        fedprox_mu = 0.0

    os.makedirs(FLConfig.RESULTS_DIR, exist_ok=True)
    tag = "%s_s%d_agg%d_p%.2f_mu%.4f" % (
        args.mode,
        args.seed,
        agg_interval if fed_mode else 0,
        participation_ratio if fed_mode else 0.0,
        fedprox_mu if fed_mode else 0.0,
    )
    csv_path = os.path.join(FLConfig.RESULTS_DIR, "metrics_%s.csv" % tag)
    chart_path = os.path.join(FLConfig.RESULTS_DIR, "Performance_Chart_%s.png" % tag)

    env = MEC(
        FLConfig.N_UE,
        FLConfig.N_EDGE,
        FLConfig.N_TIME,
        FLConfig.N_COMPONENT,
        FLConfig.MAX_DELAY,
    )

    ue_RL_list = []
    for _ue in range(FLConfig.N_UE):
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
                fedprox_mu=fedprox_mu if fed_mode else 0.0,
            )
        )

    train_fl(
        ue_RL_list,
        env,
        num_episode,
        fed_mode=fed_mode,
        fedprox_mu=fedprox_mu,
        agg_interval=agg_interval,
        participation_ratio=participation_ratio,
        csv_path=csv_path,
        chart_path=chart_path,
        rng=rng,
    )
    print("Wrote:", csv_path)
    print("Chart:", chart_path)


if __name__ == "__main__":
    main()
