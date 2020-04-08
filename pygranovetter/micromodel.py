import os
import networkx as nx
import numpy as np
import pickle
import random

SEED = 0
np.random.seed(SEED)
random.seed(SEED)

class Micromodel():

    
    def __init__(self, number_of_nodes, average_degree, micro_threshold,
                 potentially_active=None):

        p = average_degree / (number_of_nodes - 1)
        if not potentially_active:
            potentially_active = number_of_nodes

        self._N = number_of_nodes
        self._average_degree = average_degree
        self._p = p
        self._micro_threshold = micro_threshold
        self._potentially_active = potentially_active
        
        self._results = {"number_of_nodes": number_of_nodes,
                         "average_degree": average_degree,
                         "micro_threshold": micro_threshold,
                         "potentially_active": potentially_active,
                         "data": {}}

    @classmethod
    def load(cls, input_file):

        data = pickle.load(open(input_file, "rb"))
        number_of_nodes = data["number_of_nodes"]
        average_degree = data["average_degree"]
        micro_threshold = data["micro_threshold"]
        potentially_active = data["potentially_active"]

        model = Micromodel(number_of_nodes=number_of_nodes,
                             average_degree=average_degree,
                             micro_threshold=micro_threshold,
                             potentially_active=potentially_active)
        model._results = data

        return model


    def _one_run(self, certainly_active):
        
        if certainly_active not in self._results["data"].keys():
            self._results["data"][certainly_active] = []

        if not certainly_active:
            self._results["data"][certainly_active].append([0])
            return 
        
        # Use fast_gnp_random_graph since it performs better than
        # erdos_renyi_graph
        network = nx.fast_gnp_random_graph(n=self._N, p=self._p)#, seed=SEED)
        degree = network.degree()

        potentially_active_nodes = np.random.choice(self._N, 
                                                    self._potentially_active,
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

        self._results["data"][certainly_active].append(currently_active)

  
    def run(self, certainly_active, n_runs=1):
        
        assert type(certainly_active) is not float

        if type(certainly_active) is int: 
            certainly_active = [ certainly_active ]

        total_runs = len(certainly_active) * n_runs
        current_run = 1

        for _c in certainly_active:
            for _ in range(n_runs):
                progress = current_run / total_runs * 100
                print("Running: {:5.2f} % done".format(progress), end="\r")
                self._one_run(_c)
                current_run += 1


    def print_results(self, certainly_active):
        for result in self._results["data"][certainly_active]:
            print(result)


    def aggregate(self):

        data = self._results["data"]
        certainly_active = list(data.keys())
        certainly_active.sort()
        mean = [np.mean([d[-1] for d in data[c]]) for c in certainly_active]
        std = [np.std([d[-1] for d in data[c]]) for c in certainly_active]
    
        return np.stack((certainly_active, mean, std), axis=1)


    def emergent_threshold_distribution(self, which="equilibrium", max_len=None):
        
        data = self._results["data"]
        pot = self._potentially_active

        all_x = []
        all_y = []

        if which == "equilibrium":
            for cert, entries in data.items():
                for entry in entries:
                    if len(entry) > 1:
                        all_x.append(entry[0])
                        all_y.append((entry[1] - cert) / (pot - cert))
                    if len(entry) > 2:
                        all_x.append(entry[-2])
                        all_y.append((entry[-1] - cert) / (pot - cert))
    
            all_x = np.array(all_x) / self._N
            all_y = np.array(all_y)

        if which == "off_equilibrium":
            for cert, entries in data.items():
                for entry in entries:
                    if len(entry) > 3:
                        entry = np.array(entry)
                        all_x.append(entry[1:-2])
                        all_y.append((entry[2:-1] - cert) / (pot - cert))
            
            if len(all_x):
                all_x = np.concatenate(all_x) / self._N
                all_y = np.concatenate(all_y)

        if max_len and len(all_x):
            choice = np.random.choice(len(all_x), min(max_len, len(all_x)), replace=False)
            all_x = all_x[choice]
            all_y = all_y[choice]

        return all_x, all_y



    def save(self, output_folder="./", output_file=None):
      
        if "/" in output_file:
            output_folder, output_file = output_file.rsplit("/", 1)

        if output_folder[-1] != "/":
            output_folder += "/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if output_file is None:
            output_file = "granovetter_micro_N{0}_K{1}_P{2}_T{3}.p".format(
                self._N, self._average_degree, self._potentially_active,
                self._micro_threshold)

        pickle.dump(self._results, open(output_folder+output_file, "wb"))

    
if __name__ == "__main__":
    print("Potentially active<N")
    m = Micromodel2(number_of_nodes=1000, average_degree=20,
                    micro_threshold=0.25, potentially_active=900)
    m.run(np.arange(0, 1000, 100), 5)
    m.print_results(100)

    print(m.aggregate())
