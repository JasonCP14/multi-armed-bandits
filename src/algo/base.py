from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.stats import bernoulli, norm

from src.algo.util import get_highest_mean, get_second_highest_mean
from src.arm import Arm
from src.metrics import Metrics


class BaseAlgo(ABC):
    """ The Base Top Two Abstract Class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        is_top_two (bool): The method that the algo is using, Top Two or Normal.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

    def __init__(self, arms: List[Arm], is_top_two: bool, confint: float, max_iters: int = 2000):
        self.arms = arms
        self.is_top_two = is_top_two
        self.confint = confint
        self.max_iters = max_iters
        self.metrics = Metrics(arms)
        self.beta = 0.5

    def run(self) -> dict:
        """ Runs the method. 
        
        Returns:
            dict: The iteration count and metrics of the current instance.
        """

        print("Prior Distribution Initialization:")
        for arm in self.arms:   
            print(f"Arm {arm.id}: miu = {arm.miu}, sigma^2 = {arm.sigma_sqr}")

        is_identified = False
        for i in range(1, self.max_iters+1):
            leader = self.get_leader()

            if self.is_top_two:
                challenger = self.get_challenger(leader)
                chosen_arm = (challenger, leader)[bernoulli.rvs(self.beta)]
            else:
                chosen_arm = leader

            reward = chosen_arm.pull()
            self.update(chosen_arm, reward)
            self.metrics.update(chosen_arm)
            prob = self.get_optimal_prob()
 
            if (prob > self.confint) and not is_identified:
                minimum_iter, optimal_id, optimal_prob = i, get_highest_mean(self.arms).id, prob
                is_identified = True
                final_miu = list(map(lambda x: x.miu, self.arms))
                final_sigma_sqr = list(map(lambda x: x.sigma_sqr, self.arms))
                final_pulls = list(map(lambda x: x.num_pulls, self.arms))
        
        print("Final Iteration Posterior Distribution:")
        if is_identified:
            for i in range(len(self.arms)):   
                print(f"Arm {i+1}: miu = {final_miu[i]}, sigma^2 = {final_sigma_sqr[i]}, num_pulls = {final_pulls[i]}")
            print(f"After {minimum_iter} iterations, the best arm is arm {optimal_id}, with p = {optimal_prob}\n")
        else:
            for arm in self.arms:   
                print(f"Arm {arm.id}: miu = {arm.miu}, sigma^2 = {arm.sigma_sqr}, num_pulls = {arm.num_pulls}")
            print(f"After {self.max_iters} iterations, the best arm is not identified\n")
            minimum_iter = np.NaN

        self.metrics.finalize()
        results = {
            "final_iter": minimum_iter,
            "fail": 0 if is_identified else 1,
            "pe": self.metrics.pe,
            "sr": self.metrics.sr,
            "cr": self.metrics.cr
        }

        return results
    
    def update(self, arm: Arm, reward: float) -> None:
        """ Updates the chosen arm according to the pulled reward.

        Args:
            arm (Arm): The arm to be updated.
            reward (float): The reward pulled from the true distribution.
        """

        arm.miu = (arm.miu/arm.sigma_sqr + reward/arm.variance) / (1/arm.sigma_sqr + 1/arm.variance)
        arm.sigma_sqr = 1/(1/arm.sigma_sqr + 1/arm.variance)

    def get_optimal_prob(self) -> float:
        """ Gets the probability that the current best arm is the optimal arm.

        Returns:
            float: Probability of optimality.
        """

        first = get_highest_mean(self.arms)
        second = get_second_highest_mean(self.arms)

        x = -(first.miu-second.miu) / np.sqrt(first.sigma_sqr+second.sigma_sqr)
        return 1 - norm.cdf(x)

    @abstractmethod
    def get_leader(self) -> Arm:
        """ Gets the leader based on the algo sampling.

        Returns:
            Arm: The leader.
        """

        pass

    @abstractmethod
    def get_challenger(self, leader: Arm) -> Arm:
        """ Gets the challenger based on the algo sampling.

        Args:
            leader (Arm): The leader to challenge.

        Returns:
            Arm: The challenger.
        """

        pass
