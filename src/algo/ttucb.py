import math

import numpy as np

from src.algo.base import BaseAlgo
from src.algo.util import w_bar, get_transportation_cost
from src.arm import Arm


class TTUCB(BaseAlgo):
    """ The Top Two Upper Confidence Bound method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        is_top_two (bool): The method that the algo is using, Top Two or Normal.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

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

        challenger = min(filter(lambda x: x is not leader, self.arms), key = lambda arm: get_transportation_cost(leader, arm))

        return challenger
