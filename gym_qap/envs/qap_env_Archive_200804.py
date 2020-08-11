import numpy as np
from itertools import permutations
import gym
from gym import error, spaces, utils, spaces
from gym.utils import seeding
import math 
from numpy.random import default_rng
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import pickle
import os
from gym.envs.classic_control import rendering

class qapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    
            
    
    def __init__(self):

        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        f = open(os.path.join(__location__,'qap_matrices.pkl'))
        self.DistanceMatrices, self.FlowMatrices = pickle.load(open(os.path.join(__location__,'qap_matrices.pkl'), 'rb'))
        self.instance = None
        
        while not (self.instance in self.DistanceMatrices.keys() or self.instance in self.FlowMatrices.keys() or self.instance in ['Neos-n6', 'Brewery']):
            print('Available Problem Sets:', self.DistanceMatrices.keys())
            self.instance = input('Pick a problem:')
        
        if self.instance == 'Brewery':
            self.F = np.array([[0,0,150,70,0,0,254,0,0],\
                         [0,0,22.5,0,0.25,0.06,76.4,0,0],\
                         [150,22.5,0,157.5,0,0,0,0.2,0],\
                         [70,0,157.5,0,140,0,0,50,0],\
                         [0,0.25,0,140,0,125,0,0,0],\
                         [0,0.06,0,0,125,0,110,0,0],\
                         [254,76.4,0,0,0,110,0,0.1,181.4],\
                         [0,0,0.2,50,0,0,0.1,0,0],\
                         [0,0,0,0,0,0,181.4,0,0]])  
            self.D = np.array([[0,2,4,2,2.83,4.47,4,4.47,5.66],\
                          [2,0,2,2.83,2,2.83,4.47,2,4.47],\
                          [4,2,0,4.47,2.83,2,5.66,4.47,4],\
                          [2,2.83,4.47,0,2,4,2,2.83,4.47],\
                          [2.83,2,2.83,2,0,2,2.83,2,2.83],\
                          [4.47,2.83,2,4,2,0,4.47,2.83,2],\
                          [4,4.47,5.66,2,2.83,4.47,0,2,4],\
                          [4.47,2,4.47,2.83,2,2.83,2,0,2],\
                          [5.66,4.47,4,4.47,2.83,2,4,2,0]])
            
            self.opt = np.array([5,7,8,6,3,2,4,9,1])
        elif self.instance == 'Neos-n6':

            self.D = np.array([[0,40,64,36,22,60],\
                          [40,0,41,22,36,72],\
                          [64,41,0,28,44,53],\
                          [36,22,28,0,20,50],\
                          [22,36,44,20,0,41],\
                          [60,72,53,50,41,0]])
            self.F = np.array([[0,1,1,2,0,0],\
                          [1,0,0,0,0,2],\
                          [1,0,0,0,0,1],\
                          [2,0,0,0,3,0],\
                          [0,0,0,3,0,0],\
                          [0,2,1,0,0,0]])
            self.opt = np.array([5,2,6,1,4,3]) # Cost: 313
        
        else:      
            self.D = self.DistanceMatrices[self.instance]
            self.F = self.FlowMatrices[self.instance]
        
        # Determine problem size relevant for much stuff in here:
        self.n = len(self.D[0])
        
        # Beta: prevent problems smaller than 4 and greater than 10.
        while self.n>11 or self.n<4:
            self.n = int(input('Ouch! Problem size unsuitable. Pick a number greater than 4 and smaller than 10.'))

        
        # Action space has two option:
        # 1) Define as Box with shape (1, 2) and allow values to range from 1 through self.n 
        # 2) Define as Discrete with x = 1+((n^2-n)/2) actions (one half of matrix + 1 value from diagonal) --> Omit "+1" to obtain range from 0 to x!
        # self.action_space = spaces.Box(low=-1, high=6, shape=(1,2), dtype=np.int) # Doubles complexity of the problem as it allows the identical action (1,2) and (2,1)
        self.action_space = spaces.Discrete(int((self.n**2-self.n)*0.5))
        
        # If you are using images as input, the input values must be in [0, 255] as the observation is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies.
        # For problems smaller than n = 10, the program will allow discrete state spaces; for everything above: only boxes 
        # self.observation_space = spaces.Discrete(math.factorial(self.n))
        self.observation_space = spaces.Box(low=1, high = self.n, shape=(1,self.n), dtype=np.int32)
        
        self.state = self.sampler()
        #self.states = self.statesMaker()
        self.actions = self.actionsMaker(self.n)
        #self.action = self.action_space.sample()
        self.action = None
        
    def sampler(self):
        if isinstance(self.observation_space, gym.spaces.discrete.Discrete):
            state = self.observation_space.sample()
        elif isinstance(self.observation_space, gym.spaces.box.Box):
            state = default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 
        return state
    
    def statesMaker(self): # Removed on 29.07.2020
        rng = default_rng()
        numbers = rng.choice(range(1,self.n+1), size=self.n, replace=False)
        perms = set(permutations(numbers))
        states = []
        
        if self.n < 12:
            for s in perms:
               states.append(tuple(s))
        
        return states
        
    def actionsMaker(self, x):
        actions = {}
        cnt = 0
        for idx in range(1,x):
            for idy in range(idx + 1, x+1):
                skraa = tuple([idx, idy])
                actions[cnt] = skraa
                cnt +=1        
        
        # Add idle action to dictionary
        actions[cnt] = tuple([1,1])
        return actions
                
                
    def step(self, action):
        # Create new State based on action 
        newPermutation = self.setState(self.actions[action])
        
        # Compute reward for taking action on old state:  
        # Calculate permutation matrix for new state
        P = self.permutationMatrix(newPermutation)
        
        #Deduct results from known optimal value 
        #reward = self.best if np.array_equal(newPermutation, self.opt) else self.best - 0.5*np.trace(np.dot(np.dot(self.F,P), np.dot(self.D,P.T))) #313 is optimal for NEOS n6, needs to be replaced with a dict object
        reward = 10 if np.array_equal(newPermutation, self.opt) else -1 
                    
        # Find index of new state in State Space:
        # newState = self.states.index(tuple(newPermutation))  
        newState = newPermutation
        
        #Return the step funtions observation: new State as result of action, reward for taking action on old state, done=False (no terminal state exists), None for no diagostic info
        return newState, reward, self.isTerminalState(newPermutation), None
    
    def setState(self, swap):
                   
        # Load current state as np.array (to make it indexable)
        #temp = np.array(self.states[self.state])
         
        # Swap old swap[0] and new swap[1] facilities              
        #temp[swap[0]-1], temp[swap[1]-1] = temp[swap[1]-1], temp[swap[0]-1]
        
        self.state[swap[0]-1], self.state[swap[1]-1] = self.state[swap[1]-1], self.state[swap[0]-1] 
                                       
        return self.state
     
    def permutationMatrix(self, a):
        P = np.zeros((len(a), len(a)))
        for idx,val in enumerate(a):
            P[idx][val-1]=1
        return P
    
    def isTerminalState(self, state):
        #print(state, self.opt)
        return np.array_equal(state, self.opt)
    
    def reset(self):
        new = self.observation_space.sample() # Discrete Version
        new = self.sampler()
        #print(new)
        return new

   
    def render(self):       
        #pygame.quit()
        pygame.init()
        
        font = pygame.font.SysFont('Arial', 100)
              
        # Setting up color objects
        BLUE  = (0, 0, 255)
        RED   = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        
        # Number of locations to display:
        #displayState = np.array(self.states[self.state])
        displayState = self.state
        n = len(displayState)
        
        if not self.action == None:
            loc_src = self.actions[self.action][0]
            loc_trg = self.actions[self.action][1]
        else:
            loc_src = None
            loc_trg = None
        
        # Setup a 300x300 pixel display with caption
        screen = pygame.display.set_mode((self.n * 100 ,110))
        
        #bg = pygame.image.load(r"C:\Users\User\sciebo\01_Dissertation\37_Python\QAP\gym-qap\gym_qap\envs\brewery.png")

        #INSIDE OF THE GAME LOOP
        #screen.blit(bg, (0, 0))
        
        #DISPLAYSURF.fill(WHITE)
        pygame.display.set_caption("QAP")
        
        #w = 486-373
        #h = w
        
        #if n == 9:
        # Coordinates for Brewery case:
        #    locations = [(642,308, w, h),\
        #                 (642,140, w, h),\
        #                 (508,308, w, h),\
        #                 (508,140, w, h),\
        #                 (374,308, w, h),\
        #                 (374,140, w, h),\
        #                 (240,308, w, h),\
        #                 (240,140, w, h),\
        #                 (94,218, w, h)]

        locations = []
        left = 0
        top = 0
        width = 100
        heigt = 100
        
        for i in range(1,n+1):
            
            locations.append((left, top, width, heigt))
            left = left + width
            
            if i == loc_src:
                pygame.draw.rect(screen, RED, locations[i-1])
                screen.blit(font.render(str(displayState[i-1]), True, WHITE), (locations[i-1][0]+0.25*locations[i-1][2], locations[i-1][1]+0.0*locations[i-1][3]))
            elif i == loc_trg:
                pygame.draw.rect(screen, GREEN, locations[i-1])
                screen.blit(font.render(str(displayState[i-1]), True, WHITE), (locations[i-1][0]+0.25*locations[i-1][2], locations[i-1][1]+0.0*locations[i-1][3]))
            else:
                pygame.draw.rect(screen, BLACK, locations[i-1])
                screen.blit(font.render(str(displayState[i-1]), True, WHITE), (locations[i-1][0]+0.25*locations[i-1][2], locations[i-1][1]+0.0*locations[i-1][3]))
        
        pygame.display.update()
    
        #while True:
        #    for event in pygame.event.get():
        #        if event.type == QUIT:
        #            pygame.quit()


class NEOSn6(qapEnv): # Define and register a class for each problem instance [tbd, 18.07.2020]
    def __init__(self, enable_render=True):
        super(NEOSn6, self).__init__(n = 6)       

''' 
Friedhof der Code-Schnipsel:
    
1) np.array der Länge X im Bereich A,B mit nur eindeutigen Werten herstellen:
    
    from numpy.random import default_rng
    rng = default_rng()
    numbers = rng.choice(range(A,B), size=X, replace=False)
'''