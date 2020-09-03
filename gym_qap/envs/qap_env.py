import numpy as np
from itertools import permutations
import gym
from gym import error, spaces, utils
from numpy import random as rd
import pygame
import pickle
import os
import math

class qapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}          
    
    def __init__(self, \
                 stateMode):

        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        try: 
            self.DistanceMatrices, self.FlowMatrices = pickle.load(open(os.path.join(__location__,'qap_matrices.pkl'), 'rb'))
        except:
            self.DistanceMatrices, self.FlowMatrices = pickle.load(open(os.path.join('/input/qaplib-matrices/','qap_matrices.pkl'), 'rb'))  
        
        self.instance = None
        
        while not (self.instance in self.DistanceMatrices.keys() or self.instance in self.FlowMatrices.keys() or self.instance in ['Neos-n6', 'Neos-n7', 'Brewery']):
           #print('Available Problem Sets:', self.DistanceMatrices.keys())
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
            self.best = 313      
        elif self.instance == 'Neos-n7':
            self.D = np.array([[0,35,71,99,71,75,41],\
                      [35,0,42,80,65,82,47],\
                      [71,42,0,45,49,79,55],\
                      [99,80,45,0,36,65,65],\
                      [71,65,49,36,0,31,32],\
                      [75,82,79,65,31,0,36],\
                      [41,47,55,65,32,36,0]])
            self.F = np.array([[0,2,0,0,0,0,2],\
                      [2,0,3,0,0,1,0],\
                      [0,3,0,0,0,1,0],\
                      [0,0,0,0,3,0,1],\
                      [0,1,1,0,0,0,0],\
                      [2,0,0,1,0,0,0]])
            self.opt = tuple(np.array([5,4,7,1,2,3,6])) # Cost: 470
            self.best = 470
            
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
        
        if stateMode == "Discrete":
            self.observation_space = spaces.Discrete(math.factorial(self.n))
        elif stateMode == "Box":
            self.observation_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.float32)
        
        self.states = [] if self.n > 11 else self.statesMaker()
        self.actions = self.actionsMaker(self.n)
        
        # Initialize Environment with empty state and action
        self.action = None
        self.state = None
        
        #Initialize moving target to incredibly high value. To be updated if reward obtained is smaller. 
        self.movingTargetReward = np.inf 
        
    def sampler(self):
        if isinstance(self.observation_space, gym.spaces.discrete.Discrete):
            s = self.observation_space.sample()
        elif isinstance(self.observation_space, gym.spaces.box.Box):
            s = rd.default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 
        return s
    
    def statesMaker(self): # Removed on 29.07.2020
        rng = rd.default_rng()
        numbers = rng.choice(range(1,self.n+1), size=self.n, replace=False)
        perms = set(permutations(numbers))
        states = []
        
        if self.n < 12:
            for s in perms:
                states.append(s)
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
        
        swap = self.actions[action]
        
      
        if isinstance(self.observation_space, gym.spaces.discrete.Discrete):
            if isinstance(self.state, int):
                fromState = np.array(self.states[self.state])
        else:
            fromState = np.array(self.state)
        
        fromState[swap[0]-1], fromState[swap[1]-1] = fromState[swap[1]-1], fromState[swap[0]-1]
            
        # Compute reward for taking action on old state:  
        # Calculate permutation matrix for new state
        P = self.permutationMatrix(fromState)
        
        #Deduct results from known optimal value 
        #reward = self.best if np.array_equal(fromState, self.opt) else self.best - 0.5*np.trace(np.dot(np.dot(self.F,P), np.dot(self.D,P.T))) #313 is optimal for NEOS n6, needs to be replaced with a dict object
        MHC = 0.5*np.trace(np.dot(np.dot(self.F,P), np.dot(self.D,P.T)))
        
        if self.movingTargetReward == np.inf:
            self.movingTargetReward = MHC 
        
        reward = self.movingTargetReward - MHC
        self.movingTargetReward = MHC if MHC < self.movingTargetReward else self.movingTargetReward
        
        finished = np.array_equal(fromState, self.opt)
        
        if isinstance(self.observation_space, gym.spaces.discrete.Discrete):
            newState = self.states.index(tuple(fromState))
        else:
            newState = np.array(fromState)
        #Return the step funtions observation: new State as result of action, reward for taking action on old state, done=False (no terminal state exists), None for no diagostic info
        return newState, reward, finished, None
    
    def setState(self, swap, oldState):                  
        if isinstance(self.observation_space, gym.spaces.discrete.Discrete):
            if isinstance(oldState, int):
                oldState = np.array(self.states[oldState])
        else:
            oldState = oldState
        
        oldState[0,swap[0]-1], oldState[0,swap[1]-1] = oldState[0,swap[1]-1], oldState[0,swap[0]-1] 
                                       
        return oldState if gym.spaces.discrete.Discrete else self.states.index(tuple(oldState[0]))
     
    def permutationMatrix(self, a):
        P = np.zeros((len(a), len(a)))
        for idx,val in enumerate(a):
            P[idx][val-1]=1
        return P
    
    def isTerminalState(self, state):
        #print(state, self.opt)
        return np.array_equal(state, self.opt)
    
    def reset(self):
        new = self.sampler()
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
        if isinstance(self.state, int):
            displayState = np.array(self.states[self.state])
        else:
            displayState = self.state
        
        n = len(displayState)
        
        #n = displayState.shape[1]
        if not self.action == None:
            loc_src = self.actions[self.action][0]
            loc_trg = self.actions[self.action][1]
        else:
            loc_src = None
            loc_trg = None
        
        # Setup a 300x300 pixel display with caption
        screen = pygame.display.set_mode((self.n * 100 ,110))

        pygame.display.set_caption("QAP")
        
        locations = []
        left, top, width, height = 0, 0, 100, 100
        
        for i in range(1,n+1):
            
            locations.append((left, top, width, height))
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

''' 
Friedhof der Code-Schnipsel:
    
1) np.array der Länge X im Bereich A,B mit nur eindeutigen Werten herstellen:
    
    from numpy.random import default_rng
    rng = default_rng()
    numbers = rng.choice(range(A,B), size=X, replace=False)
'''
