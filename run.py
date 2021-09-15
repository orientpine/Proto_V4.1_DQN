# 실행하기 위한 코드
# python gettingStarted.py -n swarm_ver8 -m 1  

import gym
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
#from ppo2 import PPO2
from ppo2 import PPO2
from utils.saveandload import save, load
import argparse

# 개별로 만든 환경을 import하기 위해서는 해당 Env 폴더의 setup.py가 있는 dir에서 
# pip install -e .
# 를 입력해야함
import larva_ver4

from matplotlib import animation
import matplotlib.pyplot as plt
import gym 

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[32, 32,],
                                                          vf=[32, 32,])],
                                           feature_extraction="mlp")

# mode 0 = for train, mode 1 =for use

ap = argparse.ArgumentParser()
ap.add_argument('-m','--mode',type=int,required=True,help='choose the purpose of this run')
ap.add_argument('-n','--name',type=str,required=True,help='name of Env to run')
args = vars(ap.parse_args())

mode = args['mode']
#env = BipedalWalker()
env = gym.make("larva_ver4-v0")


# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# 학습용 시간 단위
secPerFrame = 480
updatePerAction = 30
updateNum = 8000
checkTime = 10 # 확인용 시간 단위
# how to load PPO2 model
# 학습 모델에 따라 저장 방법이 달라짐
if mode == 1:
    model = PPO2.load(f'trainedModel\\PPO2_{args["name"]}')
elif mode == 0:
    #model = PPO2(MlpPolicy, env, verbose=0) # MLP (2 layers of 64)
    model = PPO2(MlpPolicy, env, verbose=0, n_steps=secPerFrame*updatePerAction) # MLP (2 layer of 64), n_steps의경우 지금 480step 당 1action이니까 약 50번의 action을보고 update를 진행함
    model.learn(total_timesteps=secPerFrame*updateNum*updatePerAction) # 1초당 240 step * 1분당 60초 * 원하는 학습시간(분)
    model.save(f'trainedModel\\PPO2_{args["name"]}')


obs = env.reset()
for i in range(240*60*checkTime):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    # dones을 check하게 되면, 실패한 것을 관찰하지 않을 수 있음!
    # episode가 끝났는지 확인함
    # 아래 코드는 실제 구현할 때는 끄기
    #if dones is True:
    #   env.reset()

env.close()

