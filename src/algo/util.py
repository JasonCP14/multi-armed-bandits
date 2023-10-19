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

def get_transportation_cost(arm_1: Arm, arm_2: Arm) -> float:
    """ Calculates transportation cost between arm 1 and arm 2.

    Args:
        arm_1 (Arm): First arm.
        arm_2 (Arm): Second arm.

    Returns:
        float: The transportation cost between the 2 arms.
    """

    weighted_miu = (arm_1.num_pulls * arm_1.miu + arm_2.num_pulls * arm_2.miu) / (arm_1.num_pulls + arm_2.num_pulls)
    w = arm_1.num_pulls * kl_div(arm_1.miu, weighted_miu, arm_1.variance) + arm_2.num_pulls * kl_div(arm_2.miu, weighted_miu, arm_2.variance)

    return w

def kl_div(miu_x: float, miu_y: float, sigma_sqr: float) -> float:
    """ Calculates the KL divergence between 2 Normal distributions.

    Args:
        miu_1 (float): First miu.
        miu_2 (float): Second miu.
        sigma_sqr (float): Common sigma^2.

    Returns:
        float: The KL divergence between the 2 Normal distributions.
    """

    return np.square(miu_x - miu_y) / (2 * sigma_sqr)

def w_bar(x: float) -> float:
    """ Calculates W_bar^-1(x).

    Returns:
        float: The result.
    """

    return x + np.log(x)

