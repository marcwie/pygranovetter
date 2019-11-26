import numpy as np
from scipy import stats


class Macromodel():

    def __init__(self, micro_threshold, average_degree, accuracy=200,
                 number_of_nodes=None):

        self._micro_threshold = micro_threshold
        self._average_degree = average_degree
        self._x = np.linspace(0, 1, accuracy)

        if number_of_nodes is None:
            number_of_nodes = average_degree * 10

        self._y = self.approximation(x=self._x, number_of_nodes=number_of_nodes) 


    def diagonal_line(self, certainly_active, potentially_active):
#        if certainly_active == potentially_active:
#            certainly_active = .9999999 * certainly_active

        x = self._x
        numerator = x - certainly_active
        denumerator = potentially_active - certainly_active
        return numerator / denumerator


    def fixed_points(self, certainly_active, potentially_active):
        """
        Compute the fixed points of the macroscopic model. 

        Parameters:
        -----------
        certainly_active : float
            The share of certainly active nodes, i.e., the intersection of the
            diagonal line with the horizontal line at R(t+1) = 0
        potentially_active : float
            The share of potentially active nodes, i.e., the intersection of the
            diagonal line with the horizontal line at (R(t+1)) - C / (P - C) = 1

        Returns: 
        --------
        2d np.ndarray
            The fixed points of the granovetter macromodel (either one or three
            values). Each row holds the x and y-coordinate of one fixed point.
            If one fixed point is returned it is a global stable fixed point.
            If three fixed points are returned the smalles and largest one are
            stable while the middle one is unstable.
        """
        x = self._x
        y = self._y
        diag = self.diagonal_line(certainly_active=certainly_active,
                                  potentially_active=potentially_active)
        roots = y - diag

        # Catch some trivial cases to account for numerical errors
        if certainly_active == 0:
            roots[0] = 0
        if potentially_active == 1:
            roots[-1] = 0

        # If y == diag a root is found
        mask = (roots == 0)

        # Also, if y - diag changes sign a root is approximately found
        mask[:-1][(roots[1:] * roots[:-1]) < 0] = 1

        fixed_points = np.append(x[mask], y[mask]).reshape(2, -1).T

        return fixed_points


    def x(self):
        return self._x


    def effective_thresholds(self):
        """
        The effective thresholds from the macro approximation.

        The approximation is based on a multinomial distribution that is again
        approximated by two Poission distributions.

        Returns:
        --------
        1d np.ndarray
        """
        return self._y


    def approximation(self, x, number_of_nodes=10000):
        """
        Compute the distribution of effective thresholds.

        The approximation is based on a multinomial distribution that is again
        approximated by two Poission distributions.

        Parameters
        ----------
        x : float 
            The value at which the effective threshold is computed. It represents
            the share of active individuals in the population at time t.  Hence, x
            should be drawn from the interval [0, 1].
        number_of_nodes : int
            The number of nodes in the approximated Erdos-Renyi network. This
            number should be large for the approximation to work as in theory
            N=infinity. The default choice of 10000 produces good results.

        Returns:
        --------
        float 
            The effective threshold, i.e., the share of active individuals at
            time t+1.
        """
        micro_threshold = self._micro_threshold
        average_degree = self._average_degree

        if (micro_threshold == 1):
            return np.ones_like(x)
        
        lambda_a = average_degree * x
        lambda_b = average_degree * (1 - x)
        
        ul_b1 = np.arange(number_of_nodes + 1)
        ul_a2 = np.floor(ul_b1 * micro_threshold / (1 - micro_threshold))
            
        b_term = np.array([stats.poisson.pmf(mu=l_b, k=ul_b1) for l_b in lambda_b])
        a_term = np.array([stats.poisson.cdf(mu=l_a, k=ul_a2) for l_a in lambda_a])
    
        y = 1 - (b_term * a_term).sum(axis=1)
        
        y[x < 0] = 0
        y[x > 1] = 1

        return y



if __name__ == "__main__":
    M = Macromodel(micro_threshold=0.5, average_degree=10)
    print(M.x(), M.effective_thresholds())
