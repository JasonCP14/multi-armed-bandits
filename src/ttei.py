import logging
import math
from typing import List

import numpy as np
from scipy.stats import bernoulli, norm

from src.arm import Arm

logger = logging.getLogger(__name__)

class TTEI:
    """ The Top Two Expected Improvement method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        beta (float): Beta hyperparameter to choose amongst the top two.
        max_iters (int): Number of iterations that the method can go through.
    """

    def __init__(self, arms: List[Arm], beta: float, max_iters: int):
        self.arms = arms
        self.beta = beta
        self.max_iters = max_iters
        
    def run(self) -> None:
        """ Runs the method. """

        for i in range(self.max_iters):
            leader = self.get_leader()
            challenger = self.get_challenger(leader)
            chosen_arm = (challenger, leader)[bernoulli.rvs(self.beta)]
            reward = chosen_arm.pull()

            print(f"Iter {i}: Arm {chosen_arm.id}, miu = {chosen_arm.miu}, sigma^2 = {chosen_arm.sigma_sqr}")
            self.update(chosen_arm, reward)

    def get_leader(self) -> Arm:
        """ Gets the leader based on the EI sampling.

        Returns:
            Arm: The leader.
        """

        target = self.get_highest_posterior_mean()
        leader, leader_value = None, -math.inf
        for arm in self.arms:
            x = (arm.miu - target.miu) / np.sqrt(arm.sigma_sqr)
            value = np.sqrt(arm.sigma_sqr) * self.f(x)
            if value > leader_value:
                leader, leader_value = arm, value

        return leader

    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the EI sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        challenger, challenger_value = None, -math.inf
        for arm in self.arms:
            if arm is not leader:
                x = (arm.miu - leader.miu) / np.sqrt(arm.sigma_sqr + leader.sigma_sqr)
                value = np.sqrt(arm.sigma_sqr + leader.sigma_sqr) * self.f(x)
                if value > challenger_value:
                    challenger, challenger_value = arm, value

        return challenger

    def update(self, arm, reward: float) -> None:
        """ Updates the chosen arm according to the pulled reward.

        Args:
            reward (float): The reward pulled from the true distribution.
        """

        arm.miu = (arm.miu/arm.sigma_sqr + reward/arm.variance) / (1/arm.sigma_sqr + 1/arm.variance)
        arm.sigma_sqr = 1/(1/arm.sigma_sqr + 1/arm.variance)

    def get_highest_posterior_mean(self) -> Arm:
        """ Gets the arm with the highest posterior mean.

        Returns:
            Arm: The arm with the highest posterior mean.
        """

        return max(self.arms, key = lambda x: x.miu)
    
    def f(self, x: float) -> float:
        """ Calculates f(x).

        Returns:
            float: The result.
        """

        return x * norm.cdf(x) + norm.pdf(x)