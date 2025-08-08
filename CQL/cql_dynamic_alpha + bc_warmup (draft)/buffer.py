import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool_)

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size

        if isinstance(state, tuple):   # gym sometimes returns (obs, info)
            state = state[0]
        if isinstance(state_, tuple):
            state_ = state_[0]

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = state_
        self.terminal_memory[idx] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = len(self)
        if max_mem == 0:
            raise ValueError("ReplayBuffer is empty.")
        batch = np.random.choice(max_mem, batch_size, replace=True)

        states  = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones   = self.terminal_memory[batch].astype(np.float32)

        return states, actions, rewards, states_, dones
