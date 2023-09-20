import logging
import math
from typing import List

import numpy as np
from scipy.stats import bernoulli

from src.algo.util import f, get_highest_mean
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

            print(self.check_stop())

    def get_leader(self) -> Arm:
        """ Gets the leader based on the TTEI sampling.

        Returns:
            Arm: The leader.
        """

        target = get_highest_mean()
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

    def check_stop(self) -> bool:
        """ Checks if the value estimates have satisfied the Chernoff's Stopping Rule.

        Returns:
            bool: Stop or not stop.
        """

        max_z = -math.inf
        target = get_highest_mean()
        for arm in self.arms:
            if arm is not target:
                total_pulls = target.num_pulls + arm.num_pulls
                weighted_average = (target.miu * target.num_pulls + arm.miu * arm.num_pulls) / total_pulls
                z = target.num_pulls * self.kl_div(target.miu, weighted_average, target.variance) + arm.num_pulls * self.kl_div(arm.miu, weighted_average, target.variance)
                max_z = max(max_z, z)
        return max_z

    def kl_div(self, miu_x, miu_y, sigma_sqr):
        return np.square(miu_x - miu_y) / (2 * sigma_sqr)
