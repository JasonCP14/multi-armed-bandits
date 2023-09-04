from src.arm import Arm
import math
import numpy as np
from typing import List
from scipy.stats import bernoulli

class ExpectedImprovement:

    def __init__(self, arms: List[Arm], beta: float, max_iters: int):
        self.arms = arms
        self.beta = beta
        self.max_iters = max_iters
        
    def run(self):
        for i in range(self.max_iters):
            leader = self.get_leader()
            challenger = self.get_challenger(leader)
            chosen_arm = (challenger, leader)[bernoulli.rvs(self.beta)]
            reward = chosen_arm.pull()
            self.update(chosen_arm, reward)

    def get_leader(self):
        target = self.get_highest_posterior_mean()
        leader, leader_value = None, -math.inf
        for arm in self.arms:
            x = (arm.miu - target.miu) / np.sqrt(arm.sigma_sqr)
            value = np.sqrt(arm.sigma_sqr) * f(x)
            if value > leader_value:
                leader, leader_value = arm, value

        return leader

    def get_challenger(self, leader):
        challenger, challenger_value = None, -math.inf
        for arm in self.arms:
            if arm is not leader:
                x = (arm.miu - leader.miu) / np.sqrt(arm.sigma_sqr + leader.sigma_sqr)
                value = np.sqrt(arm.sigma_sqr + leader.sigma_sqr) * f(x)
                if value > challenger_value:
                    challenger, challenger_value = arm, value

        return challenger

    def update(arm, reward):
        arm.miu = (arm.miu/arm.sigma_sqr + reward/arm.variance) / (1/arm.sigma_sqr + 1/arm.variance)
        arm.sigma = 1/(1/arm.sigma_sqr + 1/arm.variance)

    def get_highest_posterior_mean(self) -> Arm:
        return max(self.arms, key = lambda x: x.miu)
    
    def f(self, x):
        return # TODO