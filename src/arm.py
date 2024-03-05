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
    
    def __init__(self, id: int, distribution: str, mean: float, variance: float, params: dict = None):
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
            a = self.params["a"]
            b = self.params["b"]
            V = (a * b) / (np.square(a+b) * (a + b + 1))
            m = beta.mean(a, b, scale = np.sqrt(1 / V))
            value = beta.rvs(a, b, self.mean - m, np.sqrt(1 / V))
        elif self.distribution == "G":       
            a = self.params["a"]
            b = np.sqrt(a)
            value = gamma.rvs(a, self.mean - (a / b), 1 / b)
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