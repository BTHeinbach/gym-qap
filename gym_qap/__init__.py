# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:32:45 2020

@author: User
"""

#The init py file comes here

from gym.envs.registration import register
 
register(id='qap-v0', entry_point='gym_qap.envs:qapEnv')