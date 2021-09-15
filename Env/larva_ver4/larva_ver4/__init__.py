from gym.envs.registration import register
 
register(
    id='larva_ver4-v0',
    entry_point='larva_ver4.envs:larva_ver4Env',
    kwargs={'render': True}
)