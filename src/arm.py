from typing import List

import numpy as np
from scipy.stats import beta, gamma, norm


class Arm:
    """ The Arm class.
    
    Attributes:
        id (int): Arm ID.
        distribution (str): True distribution of the arm. "N", "B", "G".
        mean (float): True mean/loc of the arm.
        variance (float): True variance/scale of the arm.
        params (dict): True parameters of the arm. None, ["a", "b"], ["a"].
    """
    
    def __init__(self, id: int, distribution: str, mean: float, variance: float = 1, params: dict = None):
        self.id = id           
        self.distribution = distribution    
        self.mean = mean
        self.variance = variance 
        self.params = params            
        self.initialize()
        
    def initialize(self) -> None:
        """ Initializes the miu, sigma^2 and number of pulls. """

        self.num_pulls = 0  
        self.rewards = []
        self.miu = self.pull()
        self.sigma_sqr = self.variance              
    
    def pull(self) -> float:
        """ Pulls this arm to get a reward from the true distribution.

        Returns:
            float: The arm's reward.
        """
    
        std = np.sqrt(self.variance)
        if self.distribution == "N":
            value = norm.rvs(self.mean, std)   
        elif self.distribution == "B":
            value = beta.rvs(self.params["a"], self.params["b"], self.mean, std)
        elif self.distribution == "G":
            value = gamma.rvs(self.params["a"], self.mean, std)
        self.num_pulls += 1     
        self.rewards.append(value)
        
        return value
    
    def sample(self) -> float:
        """ Pulls this arm to get a reward from the true distribution.

        Returns:
            float: The arm's reward.
        """

        value = norm.rvs(self.miu, np.sqrt(self.sigma_sqr))
        
        return value