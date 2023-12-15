from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.stats import bernoulli, norm
from scipy.optimize import minimize

from src.algo.util import get_highest_mean, get_second_highest_mean
from src.arm import Arm
from src.metrics import Metrics


class BaseAlgo(ABC):
    """ The Base Top Two Abstract Class.
    
    Attributes:
        arms (List[Arm]): List of arms in the current problem.
        is_top_two (bool): The method that the algo is using, Top Two or Normal.
        is_adaptive_beta (bool): The method to choose the beta value. Adaptive or Fixed.
        confint (float): Confidence interval of optimality between 0 and 1.
        max_iters (int): Number of iterations that the method can go through.
        beta (float): Beta hyperparameter to choose amongst the top two.
    """

    def __init__(self, arms: List[Arm], is_top_two: bool, is_adaptive_beta: bool,
                 confint: float = 0.9999, max_iters: int = 1000):
        self.arms = arms
        self.is_top_two = is_top_two
        self.is_adaptive_beta = is_adaptive_beta
        self.confint = confint
        self.max_iters = max_iters
        self.metrics = Metrics(arms)
        self.beta = 0.5

    def run(self) -> dict:
        """ Runs the method. 
        
        Returns:
            dict: The iteration count and metrics of the current instance.
        """

        is_identified = False
        for i in range(1, self.max_iters+1):
            leader = self.get_leader()

            if self.is_top_two:
                if self.is_adaptive_beta and (i%10 == 1 and i>1):
                    self.update_beta()
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
        
        print("Final Iteration Posterior Distribution:")
        for arm in self.arms:   
            print(f"Arm {arm.id}: miu = {arm.miu}, sigma^2 = {arm.sigma_sqr}")

        if is_identified:
            print(f"After {minimum_iter} iterations, the best arm is arm {optimal_id}, with p = {optimal_prob}\n")
        else:
            print(f"After {self.max_iters} iterations, the best arm is not identified\n")
            minimum_iter = None

        self.metrics.finalize()
        results = {
            "final_iter": minimum_iter,
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

    def update_beta(self) -> None:
        """ Updates the beta by maximizing the objective function."""

        cur_best_arm = get_highest_mean(self.arms)
        other_arms = list(filter(lambda x: x is not cur_best_arm, self.arms))
        constants = list(map(lambda arm: np.square(arm.miu - cur_best_arm.miu), other_arms))
        print(f"mius: {list(map(lambda x: x.miu, self.arms))}")
        print(f"mius other: {list(map(lambda x: x.miu, other_arms))}")
        print(f"constants: {constants}")
        c2 = constants[0]

        def objective_function(args):
            beta, w2 = args
            return 1/beta + 1/w2
        
        def sum_to_1_constraint(args):
            beta, w2 = args
            constraint = beta + w2 - 1
            var2 = 1/w2 + 1/beta
            for k in range(1, len(other_arms)):
                ck = constants[k]
                constraint += ((c2 * beta) / (ck * beta * var2 - c2))
            return constraint
        
        def non_negative_constraint(args):
            beta, w2 = args
            constraint = []
            var2 = 1/w2 + 1/beta
            for k in range(1, len(other_arms)):
                ck = constants[k]
                constraint.append((c2 * beta) / (ck * beta * var2 - c2))
            return min(constraint)
        
        """
        def three(args):
            beta, w2 = args
            var2 = 1/w2 + 1/beta
            return (c2 * beta) / (constants[1] * beta * var2 - c2)
        
        def four(args):
            beta, w2 = args
            var2 = 1/w2 + 1/beta
            return (c2 * beta) / (constants[2] * beta * var2 - c2)
        
        def five(args):
            beta, w2 = args
            var2 = 1/w2 + 1/beta
            return (c2 * beta) / (constants[3] * beta * var2 - c2)
        """

        initial_guess = [self.beta, (1-self.beta)/len(other_arms)]

        # Define constraints
        constraints = (
            {"type": "eq", "fun": sum_to_1_constraint},
            {"type": "ineq", "fun": non_negative_constraint},
            # {"type": "ineq", "fun": three},
            # {"type": "ineq", "fun": four},
            # {"type": "ineq", "fun": five},
        )

        bounds = [(0, 1), (0, 1)]

        # Perform the optimization with constraints
        result = minimize(objective_function, initial_guess, constraints=constraints, 
                          bounds = bounds, method="trust-constr")

        # Extract the maximized values
        
        print(f"beta: {result.x[0]}")
        print(f"2: {result.x[1]}")
        print(f"3: {(c2 * result.x[0]) / (constants[1] * result.x[0] * (1/result.x[1] + 1/result.x[0]) - c2)}")
        print(f"4: {(c2 * result.x[0]) / (constants[2] * result.x[0] * (1/result.x[1] + 1/result.x[0]) - c2)}")
        print(f"5: {(c2 * result.x[0]) / (constants[3] * result.x[0] * (1/result.x[1] + 1/result.x[0]) - c2)}")
        
        optimal_beta = result.x[0] 
        self.beta = optimal_beta
        print(f"Objective: {objective_function(result.x)}")
    
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
