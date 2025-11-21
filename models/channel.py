import numpy as np

def los_probability(H, d, xi1=9.61, xi2=0.16):
    theta = (180.0/np.pi) * np.arcsin(np.clip(H / np.maximum(d, 1e-9), -1.0, 1.0))
    return 1.0 / (1.0 + xi1 * np.exp(-xi2*(theta - xi1)))

def spectral_efficiency(phi, H, dist2d, mu):
    return np.log2(1 + phi / np.maximum((H*H + dist2d*dist2d)**mu, 1e-12))

def communication_rate(wm, B, rm):
    return wm * B * rm

def phi_term(Pm, beta0, P_tilde, N0):
    return (Pm*beta0*P_tilde) / np.maximum(N0,1e-20)

def distance_3d(Pu, Pm, H):
    diff = Pu[None,:] - Pm
    d2d = np.linalg.norm(diff, axis=1)
    d3d = np.sqrt(d2d*d2d + H*H)
    return d2d, d3d
