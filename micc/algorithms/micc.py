import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from micc.utils.envs_tools import check, get_shape_from_obs_space
from micc.utils.models_tools import init, get_active_func, get_init_method
from micc.models.base.distributions import Categorical
from micc.utils.discrete_util import gumbel_softmax

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)



class MLPBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
            self.embed_norm = nn.LayerNorm(obs_dim)
        
        self.mlp = MLPLayer(
            obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
        )

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)  
        x = self.mlp(x)
        return x


class ACTLayer(nn.Module):
    def __init__(
            self, action_space, inputs_dim, initialization_method, gain, args=None
        ):
            """Initialize ACTLayer.
            Args:
                action_space: (gym.Space) action space.
                inputs_dim: (int) dimension of network input.
                initialization_method: (str) initialization method.
                gain: (float) gain of the output layer of the network.
                args: (dict) arguments relevant to the network.
            """
            super(ACTLayer, self).__init__()
            self.action_type = action_space.__class__.__name__

            action_dim = action_space.n
            self.action_out = Categorical(
                inputs_dim, action_dim, initialization_method, gain
            )

    def forward(self, x, available_actions=None, deterministic=False):
        """Compute actions and action logprobs from given input.
        Args:
            x: (torch.Tensor) input to network.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        action_distribution = self.action_out(x, available_actions)
        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )
        action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs

    def get_logits(self, x, available_actions=None):

        action_distribution = self.action_out(x, available_actions)
        action_logits = action_distribution.logits
        return action_logits

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_distribution = self.action_out(x, available_actions)
        action_log_probs = action_distribution.log_probs(action)
        if active_masks is not None:
            dist_entropy = (
                action_distribution.entropy() * active_masks.squeeze(-1)
            ).sum() / active_masks.sum()
  
        else:
            dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy, action_distribution

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, n_heads=2, att_dim=64, att_out_dim=1, soft_temperature=0.1, ):
        super(MultiHeadAttention, self).__init__()
        assert (att_dim % n_heads) == 0, "n_heads must divide att_dim"
        self.att_dim = att_dim
        self.att_out_dim = att_out_dim
        self.head_att_dim = att_dim 
        self.n_heads = n_heads
        self.temperature = 5

        self.fc_q = nn.Sequential(
            nn.Linear(dim_q, self.att_dim),
            nn.ReLU()
        ) 
        self.fc_k = nn.Linear(dim_k, self.att_dim)
        self.fc_v = nn.Linear(dim_v, self.att_dim)
        self.fc_final = nn.Linear(self.att_dim, self.att_out_dim)
        self.ac = nn.ReLU()

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        q = self.fc_q(q).view(batch_size, -1, self.head_att_dim)
        k_T = self.fc_k(k).view(batch_size, -1, self.head_att_dim).permute(0, 2, 1)
        v = v.reshape(batch_size, -1, v.shape[-1])
        alpha = F.softmax(torch.matmul(q, k_T)/k.shape[-1], dim=-1)  # shape = (batch, heads, N, N)
        result = torch.matmul(alpha, v).reshape(batch_size, -1)
        # result = v.sum(dim=1)
        # result = self.fc_final(result)  # shape = (batch, N, att_out_dim)
        # result = torch.zeros_like(result).to(result.device)
        return result


class Group_Embedding(nn.Module):
    def __init__(self, agent_embedding_dim):
        super(Group_Embedding, self).__init__()
        self.agent_embedding_dim = agent_embedding_dim
        self.group_embedding_dim = 64
        self.use_ln = False

        if self.use_ln:     # layer_norm
            self.group_embedding = nn.ModuleList([nn.Linear(self.agent_embedding_dim, self.group_embedding_dim),
                                                nn.LayerNorm(self.group_embedding_dim)])
        else:
            self.group_embedding = nn.Linear(self.agent_embedding_dim, self.group_embedding_dim)
    
    def forward(self, agent_embedding, detach=False):
        if self.use_ln:
            output = self.group_embedding[1](self.group_embedding[0](agent_embedding))
        else:
            output = self.group_embedding(agent_embedding)
        
        if detach:
            output.detach()
        output = torch.sigmoid(output)
        return output


class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()

        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_dim = obs_shape[0]
        self.base = MLPBase(args, obs_dim)
      
        act_dim = self.hidden_sizes[-1]
        self.act = ACTLayer(
            action_space,
            act_dim,
            self.initialization_method,
            self.gain,
            args,
        )

        self.W = nn.Parameter(torch.rand(obs_dim, obs_dim))


        self.to(device)

    def forward(self, obs, available_actions=None, stochastic=True):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            stochastic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)


        actor_features = self.base(obs)
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic,  
        )

        return actions



    def get_logits(self, obs, available_actions=None):
        """Get action logits from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) input to network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                      (if None, all actions available)
        Returns:
            action_logits: (torch.Tensor) logits of actions for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        return self.act.get_logits(actor_features, available_actions)


class micc:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__

        self.actor = Actor(args, obs_space, act_space, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.turn_off_grad()

    def _feat(self, obs, msg):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if msg is None:
            msg = torch.zeros(obs.shape[0], self.msg_dim, device=self.device)
        elif not torch.is_tensor(msg):
            msg = torch.as_tensor(msg, dtype=torch.float32, device=self.device)
        x = torch.cat([obs, msg], dim=-1)
        return self.backbone(x)
    
    def _feat(self, obs, msg):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if msg is None:
            msg = torch.zeros(obs.shape[0], self.msg_dim, device=self.device)
        elif not torch.is_tensor(msg):
            msg = torch.as_tensor(msg, dtype=torch.float32, device=self.device)
        x = torch.cat([obs, msg], dim=-1)
        return self.backbone(x)
    
    def get_actions(self, obs, available_actions=None, stochastic=True, msg=None):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs, available_actions, 
              stochastic)
        return actions
    
    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)

        logits = self.actor.get_logits(obs, available_actions)
        actions = gumbel_softmax(
            logits, hard=True, device=self.device
        )  # onehot actions
        logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
        return actions, logp_actions
    
    def get_similar(self,
                    agent_embeddings: torch.Tensor,
                          ind,
                          agent_id: int ):
        agent_embeddings = check(agent_embeddings).to(**self.tpdv)
        loss = self.actor.get_contrastive_loss(
            agent_embeddings,
            ind,
            agent_id
        )
        return loss


    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True
 

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False


    def soft_update(self):
        """Soft update target actor."""
        for param_target, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "\\actor_agent" + str(id) + ".pt"
        )

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "\\actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)



