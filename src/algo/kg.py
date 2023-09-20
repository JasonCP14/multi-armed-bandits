import logging
import math
from typing import List

import numpy as np

from src.arm import Arm
from src.algo.util import f

logger = logging.getLogger(__name__)

class KG:
    """ The Knowledge Gradient method class.
    
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
            chosen_arm = self.get_leader()
            # challenger = self.get_challenger(leader)
            # chosen_arm = (challenger, leader)[bernoulli.rvs(self.beta)]
            reward = chosen_arm.pull()
            self.update(chosen_arm, reward)
            print(f"Iter {i}: Arm {chosen_arm.id}, miu = {chosen_arm.miu}, sigma^2 = {chosen_arm.sigma_sqr}")

    def get_leader(self) -> Arm:
        """ Gets the leader based on the KG sampling.

        Returns:
            Arm: The leader.
        """

        leader, leader_value = None, -math.inf
        for arm in self.arms:
            target = self.get_highest_mean_exclusive(arm)
            projected_sigma_sqr = self.get_projected_sigma_sqr(arm)
            x = -np.abs(arm.miu - target.miu) / np.sqrt(arm.sigma_sqr-projected_sigma_sqr)
            value = np.sqrt(arm.sigma_sqr-projected_sigma_sqr) * f(x)
            if value > leader_value:
                leader, leader_value = arm, value

        return leader

    def update(self, arm: Arm, reward: float) -> None:
        """ Updates the chosen arm according to the pulled reward.

        Args:
            arm (Arm): The arm to be updated.
            reward (float): The reward pulled from the true distribution.
        """

        arm.miu = (arm.miu/arm.sigma_sqr + reward/arm.variance) / (1/arm.sigma_sqr + 1/arm.variance)
        arm.sigma_sqr = 1/(1/arm.sigma_sqr + 1/arm.variance)

    def get_highest_mean_exclusive(self, excluded: Arm) -> Arm:
        """ Gets the arm with the highest posterior mean exclusive of the inputted arm.

        Args:
            excluded (Arm): The excluded arm.

        Returns:
            Arm: The arm with the highest posterior mean besides the inputted arm.
        """

        return max(filter(lambda x: x is not excluded, self.arms), key = lambda x: x.miu)
    
    def get_projected_sigma_sqr(self, arm: Arm) -> float:
        """ Gets the projected sigma^2 if we are choosing this arm.

        Args:
            arm (Arm): The chosen arm.

        Returns:
            Arm: The projected sigma^2.
        """

        projected_sigma_sqr = 1/(1/arm.sigma_sqr + 1/arm.variance)
        
        return projected_sigma_sqr
    
