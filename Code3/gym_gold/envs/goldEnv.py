from gym import spaces
import gym

class GoldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):
        super(GoldEnv, self).__init__()

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
        self.state = None

        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 4个离散动作：n, e, s, w
        # 定义状态空间
        self.observation_space = spaces.Discrete(8)  # 8个离散状态

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminate_states(self):
        return self.terminate_states

    def setAction(self, s):
        self.state = s

    def step(self, action):
        state = self.state
        # 判断是否游戏结束
        if state in self.terminate_states:
            return state, 0, True, {}

        key = "%d_%s" % (state, action)
        # 查找状态迁移表
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state

        self.state = next_state

        # 获取奖励
        reward = self.rewards.get(key, 0)

        # 判断是否游戏结束
        done = next_state in self.terminate_states

        return next_state, reward, done, {}

    def reset(self):
        self.state = 1
        return self.state

    def render(self, mode='human', close=False):
        pass