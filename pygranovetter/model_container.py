import os
import numpy as np
import pickle
from .macromodel import Macromodel

class ModelContainer():

    def __init__(self, filename):

        if os.path.exists(filename):
            print("Loading models from", filename)
            models = np.load(filename, allow_pickle=True)
        else:
            print("Initializing empty container of models")
            models = {"metadata": 
                      {"accuracy": set((ACCURACY,)),
                       "micro_thresholds": set(), 
                       "average_degree": set()}, 
                      "data": {}}
            self.save_model()

        self._models = models
        self._filename = filename

    def get_model(self, accuracy, micro_threshold, average_degree,
                  verbose=False):
        
        keys = self._models["data"].keys() 
        new_key = (micro_threshold, average_degree, accuracy)

        if new_key not in keys:
            if verbose:
                print("Computing:", new_key, end="\r")
            M = Macromodel(accuracy=accuracy, 
                           average_degree=average_degree,
                           micro_threshold=micro_threshold)
            self._models["data"][new_key] = M
            self._models["metadata"]["accuracy"].add(accuracy)
            self._models["metadata"]["micro_thresholds"].add(micro_threshold)
            self._models["metadata"]["average_degree"].add(average_degree)
            self.save_model()
        else:
            M = self._models["data"][new_key] 
            if verbose:
                print(new_key, "already present. Skip.")

        return M

    def save_model(self):
        
        pickle.dump(self._models, open(self._filename, "wb"))
        
