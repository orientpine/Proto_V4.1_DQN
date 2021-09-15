import os
import math
from typing import ForwardRef
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

import time

import serial,re

samplerate = 240. # 단위 1초당 x 만큼의 step 수행
dt = 1./samplerate #dt = 1./240.
Isrender = True
IsrealTime = False
isTrain = True # 양방향용
isCom = False


class larva_ver4Env(gym.Env):
    metadata = {
        'render.modes': ['human','rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False):
        self.Forward = True
        self.UpdateCount = 0
        self.Cooling = False

        self._observation = []
        #========================================================================== action 수 변경: 원래는 65 ======================
        self.action_space = spaces.MultiDiscrete([20])
        self.observation_space = spaces.Box(np.array([ -5 , -5, -1.5708, -1.5708, -1.5708, -1.5708, -1.5708, -50, False]),
                                            np.array([  5 ,  5,  1.5708,  1.5708,  1.5708,  1.5708,  1.5708,  50, True]))# 속도, 변위, 스위치ON/OFF
        
        if (Isrender):  # render
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # used by loadURDF

        self._seed()
        ###########################
        # 통신 관련 - 학습 끝나면 연결하기`
        ###########################
        if isCom == True:
            self.ser = serial.Serial('COM3', 115200)
        

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        onTime = samplerate/2
        if self._envStepCounter%(onTime) == 0: # 0.5초마다 액션을 바꿔주기 위함
            self._action  = action

        if self._envStepCounter%(onTime*2) <= onTime: # 0.5초마다 액션을 바꿔주기 위함
            self.Cooling = False
        else :
            self.Cooling = True
        self._assign_throttle(self._action)

        # 통신용 코드 추가
        send_data = "*{}E".format(self._action)
        #print(send_data)
        ###########################
        # 통신 관련 - 학습 끝나면 연결하기
        ###########################
        if isCom == True:
            #self.ser.write(send_data.encode())
            data = str(self.ser.readline().decode()) 

        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        # 시각적인 효과를 위한 rendering용도
        if (IsrealTime):  # render
            time.sleep(dt)
        
        ################################################################
        # 제어 테스트용 - 모델 학습 이후 사용
        ################################################################
        if isTrain == False:
            if self._envStepCounter % 15000 == 0:
                if self.Forward  == True:
                    self.input = -50
                    
                else:
                    self.input = 50
                self.Forward ^= True
        #########################################################################
        self._envStepCounter += 1
        return np.array(self._observation), reward, done, {}
        
    def _reset(self):
        # 강화학습 환경 세팅용
        self._action = []
        self.vt = [0, 0, 0, 0, 0]
        self.vd = 0
        self.maxVunit = 1.5708  # ver1 기준:1.5708
        self.maxV = [self.maxVunit, self.maxVunit, self.maxVunit,
                     self.maxVunit, self.maxVunit]  # 최대 관절 가동범위는 90degree = 1.5708 rad/
       
        p.resetSimulation()
        # m/s^2 default = -9.841 ==================================================== 중력 수정: 추후 반영하기 ================================================
        p.setGravity(0, 0, -9.841)  
        p.setTimeStep(dt)  # sec
        planeId = p.loadURDF("plane.urdf")
        #  전체 길이 = 0.522, 중간위치에서 출발하기 위해 절반으로 나눔
        headStartPos = [0, 0.522/2, 0.0016]
        headStartOrientation = p.getQuaternionFromEuler([0,  0, 0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.swamId = p.loadURDF(os.path.join(
            path, "larva_model_protov4.xml"), headStartPos, headStartOrientation) #larva_model_protov4.xml 는 10배짜리 모델


        # 방향 입력을 학습하기 위한 학습용 observation
        if isTrain == True:
            ######## 단방향 용#########
            print(f"\n전진")
            self.input = 50
           ###################
            # ############################# 양방향 용 #####
            # if self.Forward == True:
            #     print(f"\n전진")
            #     self.input = 50
            # else:                
            #     print(f"\n후진")
            #     self.input = -50
            # # 방향 전환용 토글
            # self.Forward ^= True
            ################################################################

        elif isTrain == False:
            self.input = 50
            self.Forward = True
        ####################### 커스텀 조건 세팅용###########################
        self._envStepCounter = 0
        self.reward_sum = 0 # reward 주기

        #  하나의 에피소드 동안의 최대 속도         
        self.Max_reward = np.NINF
       
        # reward 계산용
        # 초기화 하기
        self.last_pos_tail = p.getLinkState(bodyUniqueId=self.swamId, linkIndex=29)[0][1] # 29번이 tail
        self.last_pos_head = p.getBasePositionAndOrientation(bodyUniqueId=self.swamId)[0][1]

        self.headVelocity = 0
        self.tailVelocity = 0
        
        self._observation = self._compute_observation()

        ############################### joint 정보 얻기!###############################
        # _link_name_to_index = {p.getBodyInfo(self.swamId)[0].decode('UTF-8'):-1,}
        # for _id in range(p.getNumJoints(self.swamId)):
        #     _joint_name = p.getJointInfo(self.swamId, _id)[1].decode('UTF-8')
        #     _childename = p.getJointInfo(self.swamId, _id)[12].decode('UTF-8')
        #     _link_name_to_index[_childename] = _id
        #     print(_id,_joint_name, _childename, )
        # print(_link_name_to_index)
        ##############################################################################
        return np.array(self._observation)

    def _assign_throttle(self, action_20):
        # action_20 = [x], 0=< x < 20 으로 나옴 0~19까지의 정수가 나옴
        # 펌웨어에는 65로 되어 있으니까 해당 코드 고치기
        dv = self.maxVunit 
        # 너무 세세한 관절 움직임은 방해가된다고 생각해서 분절을 크게 나눔
        # 각 신호마다 할당되는 동작은 아래와 같음
        # -90도, -45도, +45도 +90도
        deltav = [-1.*dv,-0.8*dv, 0.8*dv, 1.*dv] #one-hot encoding 되어 있는듯 

    
        ###################################################################
        # 각도 테스트를 위한 임의의 신호
        #print(f"현재 action:{action_20}")  

        # 5개의 관절에 들어가는 신호, 신호를 받은 관절은 [-90도, -45도, +45도 +90도] 중 하나를 선택, 그렇지 않은 관절은 평평함(=0) 신호를 받음
        
        for i in range(0,5):
            if i==action_20[0]//4 and self.Cooling == False:
                self.vt[i] = deltav[action_20[0]%4]
            else:
                self.vt[i] = 0
        # 65개일 경우!
            
        vt = self.vt
        
        # 테스트용
        #vt = [0,0,-1.57,0,0]

        # joint set이 총 5개 
        # joint set 당 joint 가 6개
        # 이중 관심있는 관절 셋만 움직이기!
        # 현재 움직일 관절 번호
        # 해당 관절 이외에는 관심없음! 인줄 알았으나 나머지 관절에 default를 넣어줘야 진짜같이 움직임
        # 관절에 property를 주기 위해!
        for _id in range(-1, p.getNumJoints(self.swamId)): # -1을 넣어서 base도 추가!
            p.changeDynamics(bodyUniqueId=self.swamId,
                            linkIndex=_id,
                            contactDamping=20,
                            contactStiffness=20,
                            #rollingFriction=0.8,
                            lateralFriction=2,
                            spinningFriction=2,
                            mass=0.00000000000015,
                            restitution=0.00000000000001,
                            )
        # 실제 관절을 움직임, 관절 세트 기준으로 움직임
        # 1 관절 세트 당 6개의 관절이 있음
        for setNumber in range(0,5):
            p.setJointMotorControlArray(bodyUniqueId=self.swamId,
                                        jointIndices=[(6*setNumber) + 0,(6*setNumber) + 1, (6*setNumber) + 2, (6*setNumber) + 3, (6*setNumber) + 4, (6*setNumber) + 5],
                                        controlMode=p.POSITION_CONTROL ,
                                        targetVelocities=[0 for _ in range(0,6)],
                                        targetPositions=[vt[setNumber]/6 for _ in range(0,6)],
                                        positionGains=[0.55 for _ in range(0,6)],
                                        velocityGains=[0.5 for _ in range(0,6)],
                                        forces=[0.000000000015 for _ in range(0,6)]
                                        )
        

    def _compute_observation(self):

        # 머리와 꼬리 위치 업데이트
        head_pos = p.getBasePositionAndOrientation(bodyUniqueId=self.swamId)[0][1]
        tail_pos = p.getLinkState(bodyUniqueId=self.swamId, linkIndex=29)[0][1]
        # 머리, 꼬리 속도 구하기
        self.headVelocity = (head_pos - self.last_pos_head)/dt 
        self.tailVelocity = (tail_pos - self.last_pos_tail)/dt 

        # 머리, 꼬리 속도를 구하기 위한 pose 업데이터
        self.last_pos_tail = tail_pos
        self.last_pos_head = head_pos

        return [self.headVelocity, self.tailVelocity, self.vt[0], self.vt[1], self.vt[2], self.vt[3], self.vt[4],self.input, self.Cooling]

    def _compute_reward(self):
        # reward 계산
        reward_offset = 0
        #input이 0이면 전진
        if self.Forward == True:
            reward_direction = 1
            reward =  self.headVelocity 
        elif self.Forward == False:
            reward_direction = -1
            reward =  self.tailVelocity

        self.final_reward = reward_direction * reward + reward_offset
        self.reward_sum += self.final_reward
        self.Max_reward = (lambda x: x if x > self.Max_reward else self.Max_reward)(self.final_reward)
        return self.final_reward

    def _compute_done(self):
        headPos, _ = p.getBasePositionAndOrientation(self.swamId) # base = head의 위치

        if self.Forward == True:
            #  head의 위치가 0.5 이상
            # headPos도 [x,y,z]가 return되니까 headPos[1]이라고 써야됨
            if self.last_pos_head > 1 or self._envStepCounter%15000 == 15000-1 or  headPos[2] > 0.3: 
                print(f'\n전진 reward 총합: {self.reward_sum}, 최고 속도: {self.Max_reward}')
                return True
            
        elif self.Forward  == False:
            # tail의 위치가 -1 이상
            if self.last_pos_tail< -1 or self._envStepCounter%15000 == 15000-1 or headPos[2] > 0.3: 
                print(f'\n후진 reward 총합: {self.reward_sum}, 최고 속도: {self.Max_reward}')
                return True
        
        else:
            return False


    def _render(self, mode='human', close=False):
        pass


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)





        