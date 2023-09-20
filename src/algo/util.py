from scipy.stats import norm
from src.arm import Arm
import numpy as np
from typing import List

def get_highest_mean(arms: List[Arm]) -> Arm:
    """ Gets the arm with the highest posterior mean.

    Args:
        arms (List[Arm]): List of arms in the current problem.

    Returns:
        Arm: The arm with the highest posterior mean.
    """

    return max(arms, key = lambda x: x.miu)

def get_second_highest_mean(arms: List[Arm]) -> Arm:
    """ Gets the arm with the second highest posterior mean.

    Args:
        arms (List[Arm]): List of arms in the current problem.

    Returns:
        Arm: The arm with the second highest posterior mean.
    """

    return max(filter(lambda x: x is not get_highest_mean(arms), arms), key = lambda x: x.miu)

def f(x: float) -> float:
    """ Calculates f(x).

    Returns:
        float: The result.
    """

    return x * norm.cdf(x) + norm.pdf(x)

def get_optimal_prob(arms: List[Arm]) -> float:
    """ Gets the probability that the current best arm is the optimal arm.

    Args:
        arms (List[Arm]): List of arms in the current problem.

    Returns:
        float: Probability of optimality.
    """

    first = get_highest_mean(arms)
    second = get_second_highest_mean(arms)

    x = -(first.miu-second.miu) / np.sqrt(first.sigma_sqr+second.sigma_sqr)
    return 1 - norm.cdf(x)