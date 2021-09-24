from gym.envs.registration import register

register(
    id='StopSkip-v0',
    entry_point='envs.gym_stopskip_v0:StopSkipEnv'
)

register(
    id='StopSkip-v1',
    entry_point='envs.gym_stopskip_v1:StopSkipEnv'
)
