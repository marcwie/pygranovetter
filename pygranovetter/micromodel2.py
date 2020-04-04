import networkx as nx
import numpy as np

SEED = 0
np.random.seed(SEED)


class Micromodel2():

    
    def __init__(self, number_of_nodes, average_degree, micro_threshold):

        p = average_degree / (number_of_nodes - 1)

        self._N = number_of_nodes
        self._p = p
        self._micro_threshold = micro_threshold
    

    def run(self, certainly_active, potentially_active=None):

        if not potentially_active:
            potentially_active = self._N

        if not certainly_active:
            return [0]
        
        # Use fast_gnp_random_graph since it performs better than
        # erdos_renyi_graph
        network = nx.fast_gnp_random_graph(n=self._N, p=self._p, seed=SEED)
        degree = network.degree()

        potentially_active_nodes = np.random.choice(self._N, 
                                                    potentially_active,
                                                    replace=False)

        initially_active_nodes = np.random.choice(potentially_active_nodes,
                                                  certainly_active,
                                                  replace=False)

        active_nodes = set()
        newly_active_nodes = set(initially_active_nodes)
        inactive_nodes = set(potentially_active_nodes)
        currently_active = []

        while newly_active_nodes:
            active_nodes.update(newly_active_nodes)
            inactive_nodes.difference_update(newly_active_nodes)
            currently_active.append(len(active_nodes))
           
            newly_active_nodes = set()
            for node in inactive_nodes:
                nbs = network.neighbors(node)
                active_neighbors = active_nodes.intersection(nbs)                
                if len(active_neighbors) > (self._micro_threshold * degree[node]):
                    newly_active_nodes.add(node)

        return currently_active


if __name__ == "__main__":
    m = Micromodel2(number_of_nodes=1000, average_degree=20,
                    micro_threshold=0.25)

    print("Potentially active<N")
    for _ in range(5):
        print(m.run(100, potentially_active=900))

    print("Potentially active=N")
    for _ in range(5):
        print(m.run(100))
