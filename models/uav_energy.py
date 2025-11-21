import numpy as np

# Rotary-wing propulsion power model (Eq. (16) in the paper)
def propulsion_power(vu, C1=79.86, C2=88.63, C3=0.114, C4=0.0001, U_tip=120.0):
    # Ensure non-negative
    vu = np.maximum(0.0, vu)
    blade_profile = C1 * (1.0 + 3.0 * (vu**2) / (U_tip**2))
    # induced term: C2 * sqrt( C3 + v^4 / 4 ) - v^2 / 2  (match paper's structure)
    induced = C2 * (np.sqrt(np.sqrt(C3 + (vu**4)/4.0) - (vu**2)/2.0))
    parasite = C4 * (vu**3)
    return blade_profile + induced + parasite

def propulsion_energy(vu, tau, **kwargs):
    return propulsion_power(vu, **kwargs) * tau
