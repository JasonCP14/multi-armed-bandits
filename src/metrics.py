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
        cr (np.ndarray): The trajectory of cumulative regret.
        pe_value (int): Current probability of error value
        pe_value (float): Current probability of error value
        pe_value (float): Current probability of error value
    """
    
    def __init__(self, arms: List[Arm]):
        self.arms = arms
        self.true_best_arm = max(arms, key = lambda x: x.mean)
        self.pe, self.sr, self.cr = np.array([]), np.array([]), np.array([])
        self.pe_value, self.sr_value, self.cr_value = 0, 0, 0

    def update(self, pulled_arm: Arm) -> None:
        """ Updates the metrics trajectories. 
        
        Args:
            pulled_arm (Arm): The arm pulled in this round.
        """

        current_best_arm = max(self.arms, key = lambda x: x.miu)
        self.update_probability_error(current_best_arm)
        self.update_simple_regret(current_best_arm)
        self.update_cumulative_regret(pulled_arm)

    def update_probability_error(self, current_best_arm: Arm) -> None:
        """ Updates the probability error trajectory.

        Args:
            current_best_arm (Arm): The arm with the highest posterior mean.
        """

        error = 0 if current_best_arm is self.true_best_arm else 1
        self.pe_value += error
        self.pe = np.append(self.pe, self.pe_value)

    def update_simple_regret(self, current_best_arm: Arm) -> None:
        """ Updates the simple regret trajectory.

        Args:
            current_best_arm (Arm): The arm with the highest posterior mean.
        """

        regret = self.true_best_arm.mean - current_best_arm.mean
        self.sr_value += regret
        self.sr = np.append(self.sr, self.sr_value)

    def update_cumulative_regret(self, pulled_arm: Arm) -> None:
        """ Updates the cumulative error trajectory.

        Args:
            pulled_arm (Arm): The arm pulled in this round.
        """

        regret = pulled_arm.mean
        self.cr_value += regret
        self.cr = np.append(self.cr, self.cr_value)

    def finalize(self):
        for i in range(len(self.pe)):
            self.pe[i] = self.pe[i] / (i+1)
            self.sr[i] = self.sr[i] / (i+1)
            self.cr[i] = (i+1) * self.true_best_arm.mean - self.cr[i]
