import logging
import math
from typing import List

import numpy as np
from scipy.stats import bernoulli

from src.algo.util import f, get_highest_mean, get_optimal_prob
from src.arm import Arm

logger = logging.getLogger(__name__)

class TTEI:
    """ The Top Two Expected Improvement method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

    def __init__(self, arms: List[Arm], confint: float = 0.9999, max_iters: int = 1000, beta: float = 0.5):
        self.arms = arms
        self.beta = beta
        self.max_iters = max_iters
        self.confint = confint
        
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

            print(f"Iter {i}: Arm {chosen_arm.id}, miu = {chosen_arm.miu}, sigma^2 = {chosen_arm.sigma_sqr}")
            self.update(chosen_arm, reward)

            prob = get_optimal_prob(self.arms)
            if prob > self.confint:
                break

        print(f"After {i} iterations, the best arm is arm {get_highest_mean(self.arms).id}, with p = {prob}")
        return i

    def get_leader(self) -> Arm:
        """ Gets the leader based on the TTEI sampling.

        Returns:
            Arm: The leader.
        """

        target = get_highest_mean(self.arms)
        leader, leader_value = None, -math.inf
        for arm in self.arms:
            x = (arm.miu - target.miu) / np.sqrt(arm.sigma_sqr)
            value = np.sqrt(arm.sigma_sqr) * f(x)
            if value > leader_value:
                leader, leader_value = arm, value

        return leader

    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the TTEI sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        challenger, challenger_value = None, -math.inf
        for arm in self.arms:
            if arm is not leader:
                x = (arm.miu - leader.miu) / np.sqrt(arm.sigma_sqr + leader.sigma_sqr)
                value = np.sqrt(arm.sigma_sqr + leader.sigma_sqr) * f(x)
                if value > challenger_value:
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

    def kl_div(self, miu_x, miu_y, sigma_sqr):
        return np.square(miu_x - miu_y) / (2 * sigma_sqr)
