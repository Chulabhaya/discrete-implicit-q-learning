import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical


class RecurrentDiscreteValueDiscreteObs(nn.Module):
    """Recurrent discrete state value network for discrete IQL with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, seq_lengths):
        """
        Calculates state values for input states.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.

        Returns
        -------
        values : tensor
            State values for input states
        """
        # Embedding layer
        x = self.embedding(states)

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        values = self.fc3(x)

        return values


class RecurrentDiscreteCriticDiscreteObs(nn.Module):
    """Recurrent discrete soft Q-network model for discrete SAC for POMDPs with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, states, seq_lengths):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input states.
        """
        # Embedding layer
        x = self.embedding(states)

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class RecurrentDiscreteActorDiscreteObs(nn.Module):
    """Recurrent discrete actor model for discrete IQL for POMDPs with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states, seq_lengths, in_hidden=None):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input states.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        # Embedding layer
        x = self.embedding(states)

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_actions(self, states, seq_lengths, in_hidden=None, epsilon=1e-6):
        """
        Calculates actions by sampling from action distributions.

        Parameters
        ----------
        states : tensor
            Input states.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        action_probs : tensor
            Probabilities for all actions possible with input states.
        log_action_probs : tensor
            Log of action probabilities, used for entropy.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(states, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs, out_hidden

    def evaluate(self, states, actions, seq_lengths, in_hidden=None, epsilon=1e-6):
        """
        Generates log probabilities for dataset actions.

        Parameters
        ----------
        states : tensor
            States or observations.
        actions : tensor
            Action.

        Returns
        -------
        log_action_probs : tensor
            Log of probability of input actions.
        """
        action_probs, _ = self.forward(states, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        log_action_probs = dist.log_prob(actions)

        return log_action_probs


class RecurrentDiscreteValue(nn.Module):
    """Recurrent discrete state value network for discrete IQL with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, seq_lengths):
        """
        Calculates state values for input states.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.

        Returns
        -------
        values : tensor
            State values for input states
        """
        # Embedding layer
        x = F.relu(self.fc1(states))

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        values = self.fc3(x)

        return values


class RecurrentDiscreteCritic(nn.Module):
    """Recurrent discrete soft Q-network model for discrete SAC for POMDPs with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, states, seq_lengths):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input states.
        """
        # Embedding layer
        x = F.relu(self.fc1(states))

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class RecurrentDiscreteActor(nn.Module):
    """Recurrent discrete actor model for discrete IQL for POMDPs with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states, seq_lengths, in_hidden=None):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input states.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        # Embedding layer
        x = F.relu(self.fc1(states))

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_actions(self, states, seq_lengths, in_hidden=None, epsilon=1e-6):
        """
        Calculates actions by sampling from action distributions.

        Parameters
        ----------
        states : tensor
            Input states.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        action_probs : tensor
            Probabilities for all actions possible with input states.
        log_action_probs : tensor
            Log of action probabilities, used for entropy.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(states, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs, out_hidden

    def evaluate(self, states, actions, seq_lengths, in_hidden=None, epsilon=1e-6):
        """
        Generates log probabilities for dataset actions.

        Parameters
        ----------
        states : tensor
            States or observations.
        actions : tensor
            Action.

        Returns
        -------
        log_action_probs : tensor
            Log of probability of input actions.
        """
        action_probs, _ = self.forward(states, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        log_action_probs = dist.log_prob(actions)

        return log_action_probs


class DiscreteCritic(nn.Module):
    """Discrete Q-network model for discrete IQL with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, states):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input states.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class DiscreteValue(nn.Module):
    """Discrete state value network for discrete IQL with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the value model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states):
        """
        Calculates state value for input state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        value : tensor
            State value for input state.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value


class DiscreteActor(nn.Module):
    """Discrete actor model for discrete IQL with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input states.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_actions(self, states, epsilon=1e-6):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Logs of action probabilities, used for entropy.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

    def evaluate(self, states, actions):
        """
        Generates log probabilities for dataset actions.

        Parameters
        ----------
        states : tensor
            States or observations.
        actions : tensor
            Action.

        Returns
        -------
        log_action_probs : tensor
            Log of probability of input actions.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        log_action_probs = dist.log_prob(actions)

        return log_action_probs


class DiscreteCriticDiscreteObs(nn.Module):
    """Discrete Q-network model for discrete IQL with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, states):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input states.
        """
        x = self.embedding(states)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class DiscreteValueDiscreteObs(nn.Module):
    """Discrete state value network for discrete IQL with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the value model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states):
        """
        Calculates state value for input state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        value : tensor
            State value for input state.
        """
        x = self.embedding(states)
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value


class DiscreteActorDiscreteObs(nn.Module):
    """Discrete actor model for discrete IQL with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input states.
        """
        x = self.embedding(states)
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_actions(self, states, epsilon=1e-6):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Logs of action probabilities, used for entropy.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

    def evaluate(self, states, actions):
        """
        Generates log probabilities for dataset actions.

        Parameters
        ----------
        states : tensor
            States or observations.
        actions : tensor
            Action.

        Returns
        -------
        log_action_probs : tensor
            Log of probability of input actions.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        log_action_probs = dist.log_prob(actions)

        return log_action_probs
