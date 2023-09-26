import numpy as np

from src.algo.base import BaseAlgo
from src.algo.util import get_transportation_cost
from src.arm import Arm


class TTTC(BaseAlgo):
    """ The Top Two Transportation Cost method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

    def get_leader(self) -> Arm:
        """ Gets the leader based on the TTTC sampling.

        Returns:
            Arm: The leader.
        """

        leader = max(self.arms, key = lambda arm: arm.sample())

        return leader

    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the TTTC sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        challenger = min(filter(lambda x: x is not leader, self.arms), key = lambda arm: get_transportation_cost(leader, arm) + np.log(arm.num_pulls))

        return challenger
