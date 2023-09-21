import logging
import math
from typing import List

import numpy as np
from scipy.stats import bernoulli

from src.algo.util import get_highest_mean, get_optimal_prob, get_transportation_cost, w_bar
from src.arm import Arm

logger = logging.getLogger(__name__)

class TTUCB:
    """ The Top Two Upper Confidence Bound method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

    def __init__(self, arms: List[Arm], confint: float = 0.9999, max_iters: int = 1000, beta: float = 0.5):
        self.arms = arms
        self.confint = confint
        self.max_iters = max_iters
        self.beta = beta
        
    def run(self) -> int:
        """ Runs the method. 
        
        Returns:
            int: The number of iterations to reach the specified confidence interval.
        """

        for i in range(self.max_iters):
            leader = self.get_leader()
            challenger = self.get_challenger(leader)
            chosen_arm = (challenger, leader)[bernoulli.rvs(self.beta)]
            reward = chosen_arm.pull()

            self.update(chosen_arm, reward)
            prob = get_optimal_prob(self.arms)
            if prob > self.confint:
                break
        
        print("Final Iteration Posterior Distribution:")
        for arm in self.arms:   
            print(f"Arm {arm.id}: miu = {arm.miu}, sigma^2 = {arm.sigma_sqr}")

        print(f"After {i} iterations, the best arm is arm {get_highest_mean(self.arms).id}, with p = {prob}\n")
        return i

    def get_leader(self) -> Arm:
        """ Gets the leader based on the TTUCB sampling.

        Returns:
            Arm: The leader.
        """

        leader, leader_value = None, -math.inf
        for arm in self.arms:
            x = 2.88 * np.log(arm.num_pulls) + 2 * np.log(2 + 1.2*np.log(arm.num_pulls)) + 2
            value = arm.miu + np.sqrt(w_bar(x)/arm.num_pulls)
            if value > leader_value:
                leader, leader_value = arm, value

        return leader

    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the TTUCB sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        challenger, challenger_value = None, math.inf
        for arm in self.arms:
            if arm is not leader:
                value = np.abs(leader.miu - arm.miu) / np.sqrt((1/leader.num_pulls) + (1/arm.num_pulls))
                if value < challenger_value:
                    challenger, challenger_value = arm, value

        return challenger

    def update(self, arm: Arm, reward: float) -> None:
        """ Updates the chosen arm according to the pulled reward.

        Args:
            arm (Arm): The arm to be updated.
            reward (float): The reward pulled from the true distribution.
        """

        arm.miu = (arm.miu/arm.sigma_sqr + reward/arm.variance) / (1/arm.sigma_sqr + 1/arm.variance)
        arm.sigma_sqr = 1/(1/arm.sigma_sqr + 1/arm.variance)
