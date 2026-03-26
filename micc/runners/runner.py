import os
import numpy as np
import torch
import torch.nn.functional as F
import setproctitle
from torch.distributions import Categorical


from micc.common.valuenorm import ValueNorm
from micc.utils.trans_tools import _t2n
from micc.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    set_seed,
    get_num_agents,
)
from micc.utils.models_tools import init_device
from micc.algorithms.micc import micc as Policy
from micc.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic
from micc.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
from micc.utils.configs_tools import init_dir, save_config, get_task_name
from comm import (
    CommunicationEncoder,
    AttentionAggregator,
    ActionPredictor,
    build_perturbed_messages,
    discrete_log_prob_from_onehot,
)


class Runner:
    """
    MICC version:
    - critic still learns Q(s,a)
    - communication only affects actor side
    """
    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)

        self.msg_dim = self.algo_args["model"].get("msg_dim", 32)
        self.comm_hidden_dim = self.algo_args["model"].get("comm_hidden_dim", 128)
        self.comm_heads = self.algo_args["model"].get("comm_heads", 4)

        self.lambda_mi = self.algo_args["algo"].get("lambda_mi", 0.1)
        self.lambda_cc = self.algo_args["algo"].get("lambda_cc", 0.05)
        self.cc_noise_std = self.algo_args["algo"].get("cc_noise_std", 0.05)
        self.cc_dropout_prob = self.algo_args["algo"].get("cc_dropout_prob", 0.1)
        self.comm_lr = self.algo_args["model"].get("comm_lr", 5e-4)

        # communication encoder and MI head
        self.comm_encoder = torch.nn.ModuleList()
        self.mi_predictor = torch.nn.ModuleList()

        for agent_id in range(self.num_agents):
            obs_dim = self.envs.observation_space[agent_id].shape[0]
            if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                act_dim = self.envs.action_space[agent_id].n
            else:
                act_dim = int(np.prod(self.envs.action_space[agent_id].shape))

            self.comm_encoder.append(
                CommunicationEncoder(
                    obs_dim=obs_dim,
                    msg_dim=self.msg_dim,
                    hidden_dim=self.comm_hidden_dim,
                ).to(self.device)
            )
            self.mi_predictor.append(
                ActionPredictor(
                    obs_dim=obs_dim,
                    msg_dim=self.msg_dim,
                    act_dim=act_dim,
                    hidden_dim=self.comm_hidden_dim,
                ).to(self.device)
            )

        self.msg_aggregator = AttentionAggregator(
            obs_dim=self.envs.observation_space[0].shape[0],
            msg_dim=self.msg_dim,
            hidden_dim=self.comm_hidden_dim,
            num_heads=self.comm_heads,
        ).to(self.device)

        comm_params = list(self.msg_aggregator.parameters())
        for m in self.comm_encoder:
            comm_params += list(m.parameters())
        for p in self.mi_predictor:
            comm_params += list(p.parameters())

        self.comm_optimizer = torch.optim.Adam(comm_params, lr=self.comm_lr)

    # =========================================================
    # helpers
    # =========================================================
    def _to_tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def _actions_to_train_tensor(self, sp_actions):
        """
        sp_actions: list/array-like, [n_agents][B][action_dim or 1]
        return: list of tensors suitable for MI loss
        """
        out = []
        for agent_id in range(self.num_agents):
            act = self._to_tensor(sp_actions[agent_id])
            if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                act = act.long().squeeze(-1)
                act = F.one_hot(act, num_classes=self.envs.action_space[agent_id].n).float()
            else:
                act = act.float()
            out.append(act)
        return out

    def encode_messages(self, sp_obs):
        """
        sp_obs: (n_agents, batch_size, obs_dim)
        returns:
            obs_t_list: list of [B, obs_dim]
            msg_list: list of [B, msg_dim]
            agg_list: list of [B, msg_dim]
        """
        obs_t_list = [self._to_tensor(sp_obs[i]) for i in range(self.num_agents)]
        msg_list = [self.comm_encoder[i](obs_t_list[i]) for i in range(self.num_agents)]
        agg_list = self.msg_aggregator(obs_t_list, msg_list)
        return obs_t_list, msg_list, agg_list

    def compute_mi_loss(self, obs_t_list, msg_list, act_t_list):
        """
        MI lower bound:
            max E log q_eta(a_i | m_i, o_i)
        optimization form here is minimization => negative log-likelihood / regression
        """
        losses = []
        for i in range(self.num_agents):
            pred = self.mi_predictor[i](obs_t_list[i], msg_list[i])
            if self.envs.action_space[i].__class__.__name__ == "Discrete":
                logp = discrete_log_prob_from_onehot(pred, act_t_list[i])
                losses.append(-logp.mean())
            else:
                losses.append(F.mse_loss(pred, act_t_list[i]))
        return torch.stack(losses).mean()

    def compute_cc_loss(self, obs_t_list, msg_list, available_actions=None):
        """
        KL( pi(.|o,M) || pi(.|o,M') )
        requires actor to support:
            get_dist(obs, msg, available_actions=None)
        """
        perturbed_msgs = build_perturbed_messages(
            msg_list,
            noise_std=self.cc_noise_std,
            dropout_prob=self.cc_dropout_prob,
        )
        perturbed_agg = self.msg_aggregator(obs_t_list, perturbed_msgs)

        cc_terms = []
        for i in range(self.num_agents):
            avail_i = None
            if available_actions is not None:
                avail_i = available_actions[i]

            dist = self.actor[i].get_dist(obs_t_list[i], msg_list[i], avail_i)
            dist_pert = self.actor[i].get_dist(obs_t_list[i], perturbed_agg[i], avail_i)

            if self.envs.action_space[i].__class__.__name__ == "Discrete":
                p = dist.probs.clamp_min(1e-8)
                q = dist_pert.probs.clamp_min(1e-8)
                kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()
            else:
                kl = torch.distributions.kl.kl_divergence(dist, dist_pert).mean()

            cc_terms.append(kl)

        return torch.stack(cc_terms).mean()

    # =========================================================
    # actor update with MICC communication
    # =========================================================
    def update_actor_with_micc(self, sp_share_obs, sp_obs, sp_available_actions, sp_valid_transition):
        obs_t_list, msg_list, agg_list = self.encode_messages(sp_obs)

        # MI + CC communication loss
        act_t_list = []
        for agent_id in range(self.num_agents):
            # placeholder current policy action for MI regularization if needed
            # but we prefer replay action labels from train()
            pass

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(np.random.permutation(self.num_agents))

        # start from frozen actions
        actions = []
        logp_actions = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                action, logp_action = self.actor[agent_id].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id] if sp_available_actions is not None else None,
                    msg=agg_list[agent_id],
                )
                actions.append(action)
                logp_actions.append(logp_action)

        for agent_id in agent_order:
            self.actor[agent_id].turn_on_grad()

            # update current agent action with communication-aware policy
            actions[agent_id], logp_actions[agent_id] = self.actor[agent_id].get_actions_with_logprobs(
                sp_obs[agent_id],
                sp_available_actions[agent_id] if sp_available_actions is not None else None,
                msg=agg_list[agent_id],
            )

            if self.state_type == "EP":
                logp_action = logp_actions[agent_id]
                actions_t = torch.cat(actions, dim=-1)
            else:
                logp_action = torch.tile(logp_actions[agent_id], (self.num_agents, 1))
                actions_t = torch.tile(torch.cat(actions, dim=-1), (self.num_agents, 1))

            value_pred = self.critic.get_values(sp_share_obs, actions_t)

            if self.algo_args["algo"].get("use_policy_active_masks", False):
                if self.state_type == "EP":
                    actor_loss = (
                        -torch.sum(
                            (value_pred - self.alpha[agent_id] * logp_action)
                            * sp_valid_transition[agent_id]
                        )
                        / sp_valid_transition[agent_id].sum()
                    )
                else:
                    valid_transition = torch.tile(
                        sp_valid_transition[agent_id], (self.num_agents, 1)
                    )
                    actor_loss = (
                        -torch.sum(
                            (value_pred - self.alpha[agent_id] * logp_action)
                            * valid_transition
                        )
                        / valid_transition.sum()
                    )
            else:
                actor_loss = -torch.mean(value_pred - self.alpha[agent_id] * logp_action)

            self.actor[agent_id].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor[agent_id].actor_optimizer.step()
            self.actor[agent_id].turn_off_grad()

            if self.algo_args["algo"]["auto_alpha"]:
                log_prob = logp_actions[agent_id].detach() + self.target_entropy[agent_id]
                alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                self.alpha_optimizer[agent_id].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[agent_id].step()
                self.alpha[agent_id] = torch.exp(self.log_alpha[agent_id].detach())

            with torch.no_grad():
                actions[agent_id], _ = self.actor[agent_id].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id] if sp_available_actions is not None else None,
                    msg=agg_list[agent_id],
                )

        if self.algo_args["algo"]["auto_alpha"]:
            self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))

    # =========================================================
    # main train
    # =========================================================
    def train(self):
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_available_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_next_available_actions,
            sp_gamma,
        ) = data

        # -----------------------------------------------------
        # 1. critic update (unchanged: still Q(s,a))
        # -----------------------------------------------------
        self.critic.turn_on_grad()

        if self.args["algo"] == "hasac":
            next_obs_t_list, next_msg_list, next_agg_list = self.encode_messages(sp_next_obs)

            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                next_action, next_logp_action = self.actor[agent_id].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                    msg=next_agg_list[agent_id],
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)

            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
            next_obs_t_list, next_msg_list, next_agg_list = self.encode_messages(sp_next_obs)

            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(
                        sp_next_obs[agent_id],
                        msg=next_agg_list[agent_id],
                    )
                )

            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )

        self.critic.turn_off_grad()

        # -----------------------------------------------------
        # 2. MICC communication update
        # -----------------------------------------------------
        obs_t_list, msg_list, agg_list = self.encode_messages(sp_obs)
        act_t_list = self._actions_to_train_tensor(sp_actions)

        mi_loss = self.compute_mi_loss(obs_t_list, msg_list, act_t_list)
        cc_loss = self.compute_cc_loss(obs_t_list, msg_list, sp_available_actions)

        comm_loss = self.lambda_mi * mi_loss + self.lambda_cc * cc_loss

        self.comm_optimizer.zero_grad()
        comm_loss.backward()
        self.comm_optimizer.step()

        # -----------------------------------------------------
        # 3. actor update
        # -----------------------------------------------------
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)

        if self.total_it % self.policy_freq == 0:
            if self.args["algo"] == "hasac":
                self.update_actor_with_micc(
                    sp_share_obs,
                    sp_obs,
                    sp_available_actions,
                    sp_valid_transition,
                )
            else:
                # deterministic actor case
                obs_t_list, msg_list, agg_list = self.encode_messages(sp_obs)

                actions = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        actions.append(
                            self.actor[agent_id].get_actions(
                                sp_obs[agent_id],
                                False,
                                msg=agg_list[agent_id],
                            )
                        )

                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))

                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()

                    actions[agent_id] = self.actor[agent_id].get_actions(
                        sp_obs[agent_id],
                        False,
                        msg=agg_list[agent_id],
                    )

                    actions_t = torch.cat(actions, dim=-1)
                    value_pred = self.critic.get_values(sp_share_obs, actions_t)
                    actor_loss = -torch.mean(value_pred)

                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()

                    with torch.no_grad():
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id],
                            False,
                            msg=agg_list[agent_id],
                        )

                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()

            self.critic.soft_update()