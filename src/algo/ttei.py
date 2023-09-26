import math

import numpy as np

from src.algo.base import BaseAlgo
from src.algo.util import f, get_highest_mean
from src.arm import Arm


class TTEI(BaseAlgo):
    """ The Top Two Expected Improvement method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        is_top_two (bool): The method that the algo is using, Top Two or Normal.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

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
