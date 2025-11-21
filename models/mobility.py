import numpy as np

class GaussMarkovMobility:
    def __init__(self, alpha=0.8, v_mean=(0,0), sigma=1.0, tau=1.0, area=(400,400), rng=None):
        self.alpha = alpha
        self.v_mean = np.array(v_mean)
        self.sigma = sigma
        self.tau = tau
        self.area = np.array(area)
        self.rng = np.random.default_rng() if rng is None else rng

    def step(self, P, V):
        noise = self.rng.normal(0, self.sigma, size=V.shape)
        V_next = self.alpha*V + (1-self.alpha)*self.v_mean + np.sqrt(1-self.alpha**2)*noise
        P_next = P + V_next*self.tau
        return P_next, V_next
