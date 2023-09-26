import numpy as np

from src.algo.base import BaseAlgo
from src.algo.util import get_transportation_cost
from src.arm import Arm


class TTEBTC(BaseAlgo):
    """ The Top Two Emprical Best - Transportation Cost method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        is_top_two (bool): The method that the algo is using, Top Two or Normal.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

    def get_leader(self) -> Arm:
        """ Gets the leader based on the TTEBTC sampling.

        Returns:
            Arm: The leader.
        """

        leader = max(self.arms, key = lambda arm: arm.miu)

        return leader

    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the TTEBTC sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        challenger = min(filter(lambda x: x is not leader, self.arms), key = lambda arm: get_transportation_cost(leader, arm) + np.log(arm.num_pulls))

        return challenger
