"""
D3QN with per-agent tf.Graph (TF1), FedProx proximal term on eval backbone only,
and ops to read/write backbone weights for federated aggregation.
"""
from collections import deque
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _is_eval_backbone_var(name):
    """eval_net l0 (LSTM), l1, l12 only — exclude Value/Advantage/Q."""
    if not name.startswith("eval_net/"):
        return False
    if "/Value/" in name or "/Advantage/" in name or "/Q/" in name:
        return False
    if "/l12/" in name:
        return True
    if "/l0/" in name:
        return True
    if "/l1/" in name:
        return True
    return False


def _is_target_backbone_var(name):
    if not name.startswith("target_net/"):
        return False
    if "/Value/" in name or "/Advantage/" in name or "/Q/" in name:
        return False
    if "/l12/" in name:
        return True
    if "/l0/" in name:
        return True
    if "/l1/" in name:
        return True
    return False


class DuelingDoubleDeepQNetworkFL:
    def __init__(
        self,
        n_actions,
        n_features,
        n_lstm_features,
        n_time,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.99,
        replace_target_iter=200,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=0.00025,
        n_lstm_step=10,
        dueling=True,
        double_q=True,
        N_L1=20,
        N_lstm=20,
        fedprox_mu=0.0,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.N_L1 = N_L1
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features
        self.fedprox_mu_default = float(fedprox_mu)

        self.memory = np.zeros(
            (
                self.memory_size,
                self.n_features + 1 + 1 + self.n_features + self.n_lstm_state + self.n_lstm_state,
            )
        )

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_net()

            t_params = tf.get_collection("target_net_params")
            e_params = tf.get_collection("eval_net_params")
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            self.eval_backbone_vars = sorted(
                [v for v in tf.trainable_variables() if _is_eval_backbone_var(v.name)],
                key=lambda v: v.name,
            )
            self.target_backbone_vars = sorted(
                [v for v in tf.global_variables() if _is_target_backbone_var(v.name)],
                key=lambda v: v.name,
            )
            if len(self.eval_backbone_vars) != len(self.target_backbone_vars):
                raise RuntimeError(
                    "eval/target backbone var count mismatch: %d vs %d"
                    % (len(self.eval_backbone_vars), len(self.target_backbone_vars))
                )

            with tf.variable_scope("fed_anchor"):
                self.anchor_vars = []
                for i, v in enumerate(self.eval_backbone_vars):
                    a = tf.get_variable(
                        "a_%d" % i,
                        shape=v.get_shape().as_list(),
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                        trainable=False,
                    )
                    self.anchor_vars.append(a)

            prox_sq = []
            for v, a in zip(self.eval_backbone_vars, self.anchor_vars):
                prox_sq.append(tf.reduce_sum(tf.square(v - a)))
            self.prox_loss = 0.5 * tf.add_n(prox_sq)

            self.mse_loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.mu_ph = tf.placeholder(tf.float32, shape=(), name="fedprox_mu")
            self.loss = self.mse_loss + self.mu_ph * self.prox_loss

            with tf.variable_scope("train"):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            self.assign_anchor_from_eval_op = [
                tf.assign(a, v) for a, v in zip(self.anchor_vars, self.eval_backbone_vars)
            ]
            self.assign_eval_backbone_from_anchor_op = [
                tf.assign(v, a) for v, a in zip(self.eval_backbone_vars, self.anchor_vars)
            ]
            self.sync_target_backbone_from_eval_op = [
                tf.assign(tv, ev)
                for tv, ev in zip(self.target_backbone_vars, self.eval_backbone_vars)
            ]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.assign_anchor_from_eval_op)

        self.reward_store = []
        self.action_store = []
        self.delay_store = []
        self.energy_store = []

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = []

        with self.graph.as_default():
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)

    def _build_net(self):
        def build_layers(s, lstm_s, c_names, n_l1, n_lstm, w_initializer, b_initializer):
            with tf.variable_scope("l0"):
                lstm_dnn = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm)
                lstm_dnn.zero_state(self.batch_size, tf.float32)
                lstm_output, _ = tf.nn.dynamic_rnn(lstm_dnn, lstm_s, dtype=tf.float32)
                lstm_output_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, n_lstm])

            with tf.variable_scope("l1"):
                w1 = tf.get_variable(
                    "w1",
                    [n_lstm + self.n_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b1 = tf.get_variable("b1", [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(tf.concat([lstm_output_reduced, s], 1), w1) + b1)

            with tf.variable_scope("l12"):
                w12 = tf.get_variable(
                    "w12", [n_l1, n_l1], initializer=w_initializer, collections=c_names
                )
                b12 = tf.get_variable("b12", [1, n_l1], initializer=b_initializer, collections=c_names)
                l12 = tf.nn.relu(tf.matmul(l1, w12) + b12)

            if self.dueling:
                with tf.variable_scope("Value"):
                    w2 = tf.get_variable(
                        "w2", [n_l1, 1], initializer=w_initializer, collections=c_names
                    )
                    b2 = tf.get_variable("b2", [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l12, w2) + b2
                with tf.variable_scope("Advantage"):
                    w2 = tf.get_variable(
                        "w2",
                        [n_l1, self.n_actions],
                        initializer=w_initializer,
                        collections=c_names,
                    )
                    b2 = tf.get_variable(
                        "b2", [1, self.n_actions], initializer=b_initializer, collections=c_names
                    )
                    self.A = tf.matmul(l12, w2) + b2
                with tf.variable_scope("Q"):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
            else:
                with tf.variable_scope("Q"):
                    w2 = tf.get_variable(
                        "w2", [n_l1, self.n_actions], initializer=w_initializer, collections=c_names
                    )
                    b2 = tf.get_variable(
                        "b2", [1, self.n_actions], initializer=b_initializer, collections=c_names
                    )
                    out = tf.matmul(l1, w2) + b2
            return out

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")
        self.lstm_s = tf.placeholder(
            tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name="lstm1_s"
        )
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name="Q_target")
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name="s_")
        self.lstm_s_ = tf.placeholder(
            tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name="lstm1_s_"
        )

        with tf.variable_scope("eval_net"):
            c_names, n_l1, n_lstm, w_initializer, b_initializer = (
                ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES],
                self.N_L1,
                self.N_lstm,
                tf.random_normal_initializer(0.0, 0.3),
                tf.constant_initializer(0.1),
            )
            self.q_eval = build_layers(
                self.s, self.lstm_s, c_names, n_l1, n_lstm, w_initializer, b_initializer
            )

        with tf.variable_scope("target_net"):
            c_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(
                self.s_, self.lstm_s_, c_names, n_l1, n_lstm, w_initializer, b_initializer
            )

    def get_backbone_weights(self):
        """List of numpy arrays (eval backbone), fixed order."""
        with self.graph.as_default():
            return self.sess.run(self.eval_backbone_vars)

    def set_backbone_weights(self, weights_list):
        """Assign eval backbone from list of numpy arrays (same order as get_backbone_weights)."""
        if len(weights_list) != len(self.eval_backbone_vars):
            raise ValueError("backbone length mismatch")
        ops = [tf.assign(v, w) for v, w in zip(self.eval_backbone_vars, weights_list)]
        with self.graph.as_default():
            self.sess.run(ops)

    def set_anchor_from_backbone_list(self, weights_list):
        """Set FedProx anchors w^(t) (typically global broadcast)."""
        if len(weights_list) != len(self.anchor_vars):
            raise ValueError("anchor length mismatch")
        ops = [tf.assign(a, w) for a, w in zip(self.anchor_vars, weights_list)]
        with self.graph.as_default():
            self.sess.run(ops)

    def sync_anchor_from_current_eval(self):
        """After loading eval backbone, mirror anchors (round start)."""
        with self.graph.as_default():
            self.sess.run(self.assign_anchor_from_eval_op)

    def sync_target_backbone_from_eval(self):
        with self.graph.as_default():
            self.sess.run(self.sync_target_backbone_from_eval_op)

    def reinitialize_eval_backbone_random(self, stddev=0.1):
        """Randomize eval backbone (and mirror to target backbone + anchors)."""
        with self.graph.as_default():
            ops = [
                tf.assign(v, tf.random_normal(v.get_shape().as_list(), stddev=stddev))
                for v in self.eval_backbone_vars
            ]
            self.sess.run(ops)
            self.sess.run(self.sync_target_backbone_from_eval_op)
            self.sess.run(self.assign_anchor_from_eval_op)

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            lstm_observation = np.array(self.lstm_history)
            actions_value = self.sess.run(
                self.q_eval,
                feed_dict={
                    self.s: observation,
                    self.lstm_s: lstm_observation.reshape(
                        1, self.n_lstm_step, self.n_lstm_state
                    ),
                },
            )
            self.store_q_value.append({"observation": observation, "q_value": actions_value})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(1, self.n_actions)
        return action

    def learn(self, fedprox_mu=None):
        mu = self.fedprox_mu_default if fedprox_mu is None else float(fedprox_mu)

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        batch_memory = self.memory[sample_index, : self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii, jj, :] = self.memory[
                    sample_index[ii] + jj, self.n_features + 1 + 1 + self.n_features :
                ]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features :],
                self.lstm_s_: lstm_batch_memory[:, :, self.n_lstm_state :],
                self.s: batch_memory[:, -self.n_features :],
                self.lstm_s: lstm_batch_memory[:, :, self.n_lstm_state :],
            },
        )
        q_eval = self.sess.run(
            self.q_eval,
            {
                self.s: batch_memory[:, : self.n_features],
                self.lstm_s: lstm_batch_memory[:, :, : self.n_lstm_state],
            },
        )
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, : self.n_features],
                self.lstm_s: lstm_batch_memory[:, :, : self.n_lstm_state],
                self.q_target: q_target,
                self.mu_ph: mu,
            },
        )

        self.epsilon = (
            self.epsilon + self.epsilon_increment
            if self.epsilon < self.epsilon_max
            else self.epsilon_max
        )
        self.learn_step_counter += 1
        return self.cost

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
        fog_energy = 0
        for i in range(len(energy3)):
            if energy3[i] != 0:
                fog_energy = energy3[i]
        idle_energy = 0
        for i in range(len(energy4)):
            if energy4[i] != 0:
                idle_energy = energy4[i]
        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy

    def reset_personalized_head_and_memory(self, reinit_heads=True):
        """Cold-start: clear replay; optionally re-randomize Value/Advantage variables."""
        self.memory[:] = 0.0
        self.memory_counter = 0
        self.lstm_history.clear()
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))
        if not reinit_heads:
            return
        with self.graph.as_default():
            ops = []
            for v in tf.trainable_variables():
                if "/Value/" in v.name or "/Advantage/" in v.name:
                    if "eval_net" in v.name:
                        ops.append(
                            tf.assign(v, tf.random_normal(v.get_shape().as_list(), stddev=0.1))
                        )
            if ops:
                self.sess.run(ops)
            self.sess.run(self.replace_target_op)
