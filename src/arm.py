import numpy as np

class Arm:
    """ The Arm class 
    
    Args:
        mean (float): True mean of the arm's distribution
        var (float): True variance of the arm's distribution
    """
    
    def __init__(self, mean: float):                
        self.mean = mean       
        self.variance = 1            
        self.initialize()
        
    def initialize(self) -> None:
        """ Initialize the miu, sigma and number of pulls """

        self.miu = 0 
        self.sigma_sqr = 1      # TODO: change              
        self.num_pulls = 0  
    
    def pull(self) -> float:
        """ Pulls this arm to get a reward from the true distribution

        Returns:
            float: The arm's reward
        """

        value = np.random.randn(self.mean, np.sqrt(self.variance))   
        self.num_pulls += 1     
        
        return value