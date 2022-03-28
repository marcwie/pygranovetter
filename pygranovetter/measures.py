import numpy as np


def criticality(branches, certainly, potential):
    """
    Compute criticality.

    Parameters:
    -----------
    branches : 2d numpy array
        The three branches of the bifurcation diagram. Each column represents one branch (lower,
        middle/unstable, upper)
    certainly : 1d numpy array
        The values of certainly_active shares at which the bifurcation diagram is computed. Must
        have the same length as 'branches'
    potential : float
        The share of potentially active nodes

    Returns: 
    --------
    1d np.ndarray
        The criticality of the system at each value of certainly
    """
   
    criticality = np.zeros(branches.shape[0]) * np.nan
    
    only_lower = np.isnan(branches[:, 1:]).all(axis=1)
    criticality[only_lower] = 0
    
    both_exist = np.isfinite(branches[:, 1])
    criticality[both_exist] = (potential - branches[both_exist, 1]) / (potential - certainly[both_exist])

    only_upper = np.isnan(branches[:, :2]).all(axis=1)
    criticality[only_upper] = 1
    
    return criticality
