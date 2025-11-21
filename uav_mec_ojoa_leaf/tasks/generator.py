import numpy as np

class TaskGenerator:
    """
    Configurable task generator.

    Parameters
    ----------
    rng : np.random.Generator or None
        Random generator.
    D_range : tuple(float,float)
        Range of input data sizes D (bits)
    eta_range : tuple(float,float)
        Range of CPU cycles per bit
    Tmax : float
        Deadline for each task
    """

    def __init__(self, rng=None,
                 D_range=(1e5, 1e6),
                 eta_range=(500, 1500),
                 Tmax=1.0):

        self.rng = np.random.default_rng() if rng is None else rng
        self.D_range = D_range
        self.eta_range = eta_range
        self.Tmax_val = Tmax

    def generate(self, M):
        """
        Generate tasks for M users.
        Returns:
            D : array of size M
            eta : array of size M
            Tmax : array of size M
        """
        D = self.rng.uniform(self.D_range[0], self.D_range[1], size=M)
        eta = self.rng.uniform(self.eta_range[0], self.eta_range[1], size=M)
        Tmax = np.full(M, self.Tmax_val)
        return D, eta, Tmax
