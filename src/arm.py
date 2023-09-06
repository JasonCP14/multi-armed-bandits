import numpy as np
from scipy.stats import norm

class Arm:
    """ The Arm class.
    
    Attributes:
        id (int): Arm ID
        mean (float): True mean of the arm's distribution.
        variance (float): True variance of the arm's distribution.
    """
    
    def __init__(self, id: int, mean: float, variance: float = 1):
        self.id = id                
        self.mean = mean       
        self.variance = variance          
        self.initialize()
        
    def initialize(self) -> None:
        """ Initializes the miu, sigma^2 and number of pulls. """

        self.num_pulls = 0  
        self.miu = self.pull()
        self.sigma_sqr = self.variance              
        
    
    def pull(self) -> float:
        """ Pulls this arm to get a reward from the true distribution.

        Returns:
            float: The arm's reward.
        """

        value = norm.rvs(self.mean, np.sqrt(self.variance))   
        self.num_pulls += 1     
        
        return value