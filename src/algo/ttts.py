from src.algo.base import BaseAlgo
from src.arm import Arm


class TTTS(BaseAlgo):
    """ The Top Two Thompson Samping method class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        is_top_two (bool): The method that the algo is using, Top Two or Normal.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

    def get_leader(self) -> Arm:
        """ Gets the leader based on the TTTS sampling.

        Returns:
            Arm: The leader.
        """

        leader = max(self.arms, key = lambda arm: arm.sample())

        return leader

    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the TTTS sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        challenger = leader
        while challenger is leader:
            challenger = max(filter(lambda x: x is not leader, self.arms), key = lambda arm: arm.sample())

        return challenger
