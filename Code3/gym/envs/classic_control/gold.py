import gym
from gym import spaces
import numpy as np

class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        super(GridEnv, self).__init__()
        self.states = [1, 2, 3, 4, 5, 6, 7, 8]
        self.terminate_states = {6: 1, 7: 1, 8: 1}
        self.actions = ['n', 'e', 's', 'w']
        self.rewards = {
            '1_s': -1.0,
            '3_s': 1.0,
            '5_s': -1.0
        }
        self.t = {
            '1_s': 6,
            '1_e': 2,
            '2_w': 1,
            '2_e': 3,
            '3_s': 7,
            '3_w': 2,
            '3_e': 4,
            '4_w': 3,
            '4_e': 5,
            '5_s': 8,
            '5_w': 4
        }
        self.gamma = 0.8
        self.viewer = None
        self.state = self.states[0]  # 初始化为第一个状态

        # 定义动作空间和观测空间
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(len(self.states))

    def step(self, action):
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        key = f"{state}_{self.actions[action]}"
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state
        is_terminal = next_state in self.terminate_states
        r = self.rewards.get(key, 0.0)
        return next_state, r, is_terminal, {}

    def reset(self):
        self.state = self.states[int(np.random.rand() * len(self.states))]
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        # 这里可以添加渲染逻辑
        return