from typing import List

import numpy as np

from src.arm import Arm


class Metrics:
    """ The Metrics class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        true_best_arm (Arm): The arm with the highest true mean.
        pe (np.ndarray): The trajectory of probability of error.
        sr (np.ndarray): The trajectory of simple regret.
    """
    
    def __init__(self, arms: List[Arm]):
        self.arms = arms
        self.true_best_arm = max(arms, key = lambda x: x.mean)
        self.pe = np.array([])
        self.sr = np.array([])

    def update(self) -> None:
        """ Updates the metrics trajectories. """

        current_best_arm = max(self.arms, key = lambda x: x.miu)
        self.update_probability_error(current_best_arm)
        self.update_simple_regret(current_best_arm)

    def update_probability_error(self, current_best_arm: Arm) -> None:
        """ Updates the probability error trajectory.

        Args:
            current_best_arm (Arm): The arm with the highest posterior mean.
        """

        error = 0 if current_best_arm is self.true_best_arm else 1
        if self.pe.size:
            error = (self.pe[-1] * self.pe.size + error) / (self.pe.size + 1)
        
        self.pe = np.append(self.pe, error)

    def update_simple_regret(self, current_best_arm: Arm) -> None:
        """ Updates the simple error trajectory.

        Args:
            current_best_arm (Arm): The arm with the highest posterior mean.
        """

        regret = self.true_best_arm.mean - current_best_arm.miu
        if self.sr.size:
            regret = (self.sr[-1] * self.sr.size + regret) / (self.sr.size + 1)
        
        self.sr = np.append(self.sr, regret)