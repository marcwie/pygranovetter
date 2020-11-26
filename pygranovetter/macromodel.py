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
        if certainly_active == potentially_active:
            certainly_active = .99999999999 * certainly_active

        x = self._x
        numerator = x - certainly_active
        denumerator = potentially_active - certainly_active
        
        if denumerator == 0:
            denumerator = 1E-16

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

        if certainly_active > potentially_active:
            certainly_active = potentially_active

        diag = self.diagonal_line(certainly_active=certainly_active,
                                  potentially_active=potentially_active)
        roots = y - diag

        # Catch some trivial cases to account for numerical errors
        #if certainly_active == 0:
        #    roots[0] = 0

        #if potentially_active == 1:
        #    roots[-1] = 0

        # If y == diag a root is found
        mask = (roots == 0)

        # Also, if y - diag changes sign a root is approximately found
        mask[:-1][(roots[1:] * roots[:-1]) < 0] = 1

        fixed_points = np.append(x[mask], y[mask]).reshape(2, -1).T
       
        # Catch case when only two fixed points exist. In that case (0, 0) is
        # an unstable fixed points that only exists if certainly_active = 0 and
        # vanishes as soon  as certainly_active > 0. We can safely ignore this
        # point
        if (len(fixed_points) == 2) and (fixed_points[0] == 0).all():
            fixed_points = np.array([fixed_points[1]])

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
        
        y[x <= 0] = 0
        y[x > 1] = 1

        return y


    def bifurcation_diagram(self, certainly_active, potentially_active,
                            fill=False):
   
        vary_certainly = type(certainly_active) == np.ndarray
        vary_potential = type(potentially_active) == np.ndarray
    

        if not vary_certainly and not vary_potential:
            print("Either certainly or potentially active must be numpy 1d array")
            print(type(certainly_active))
            print(type(potentially_active))
            return 

        if vary_certainly:
            x = certainly_active
            y = potentially_active
        else:
            x = potentially_active
            y = certainly_active
    
        lower_branch = []
        middle_branch = []
        upper_branch = []
    
        for _x in x:
            if vary_certainly:
                fixed_points = self.fixed_points(certainly_active=_x, potentially_active=y)
            else:
                fixed_points = self.fixed_points(certainly_active=y, potentially_active=_x)

            fp_x = fixed_points[:, 0]
            if len(fp_x) == 1 and not len(middle_branch):
                lower_branch.append((_x, fp_x[0]))
            elif len(fp_x) == 3:
                lower_branch.append((_x, fp_x[0]))
                middle_branch.append((_x, fp_x[1]))
                upper_branch.append((_x, fp_x[2]))
            else:
                upper_branch.append((_x, fp_x[0]))
    
        lower_branch = np.array(lower_branch)
        middle_branch = np.array(middle_branch)
        upper_branch = np.array(upper_branch)
    
        if fill:
            branches = (lower_branch, middle_branch, upper_branch)
            full_branches = np.zeros((len(x), 4)) * np.nan
            full_branches[:, 0] = x

            for i, branch in enumerate(branches):
                if not len(branch):
                    continue
                mask = [c in branch[:, 0] for c in full_branches[:, 0]] 
                full_branches[mask, i+1] = branch[:, 1]
            
            return full_branches


        return lower_branch, middle_branch, upper_branch


if __name__ == "__main__":
    M = Macromodel(micro_threshold=0.5, average_degree=10)
    print(M.x(), M.effective_thresholds())
