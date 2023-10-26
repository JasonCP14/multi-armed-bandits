import math

import numpy as np

from src.algo.base import BaseAlgo
from src.algo.util import f, get_highest_mean
from src.arm import Arm


class TTKG(BaseAlgo):
    """ The Top Two Knowledge Gradient method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        is_top_two (bool): The method that the algo is using, Top Two or Normal.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

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

    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the TTKG sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        challenger, challenger_value = None, -math.inf
        for arm in self.arms:
            if arm is not leader:
                target = self.get_highest_mean_exclusive(arm)
                projected_sigma_sqr = self.get_projected_sigma_sqr(arm)
                x = -np.abs(arm.miu - target.miu) / np.sqrt(arm.sigma_sqr-projected_sigma_sqr)
                value = np.sqrt(arm.sigma_sqr-projected_sigma_sqr) * f(x)
                if value > challenger_value:
                    challenger, challenger_value = arm, value

        return challenger

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
    