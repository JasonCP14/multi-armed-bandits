from scipy.stats import norm
from src.arm import Arm

def get_highest_mean(self) -> Arm:
    """ Gets the arm with the highest posterior mean.

    Returns:
        Arm: The arm with the highest posterior mean.
    """

    return max(self.arms, key = lambda x: x.miu)

def f(self, x: float) -> float:
    """ Calculates f(x).

    Returns:
        float: The result.
    """

    return x * norm.cdf(x) + norm.pdf(x)