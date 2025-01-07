from gym.envs.registration import register

register(
    id='Gold-v0',
    entry_point='gym_gold.envs:GoldEnv',
)