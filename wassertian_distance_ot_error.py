import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
import pandas as pd

class EuclideanTransportMap:
    def __init__(self):
        self.source_data = None
        self.target_data = None
        self.source_weight = None
        self.target_weight = None
        self.source_data_size = 0
        self.target_data_size = 0
        self.transport_plan = None

    def loadData(self, source=None, target=None):
        if source:
            if not isinstance(source, np.ndarray):
                raise Exception("Source must be a numpy array.")
            elif source.ndim != 2:
                raise Exception("Source data dimension must be 2. First column : x, Second column : y.")
            self.source_data = source
            self.source_data_size = self.source_data.shape[0]
            print("Source data loaded.")
        if target:
            if not isinstance(target, np.ndarray):
                raise Exception("Target must be a numpy array.")
            elif target.ndim != 2:
                raise Exception("Target data dimension must be 2. First column : x, Second column : y.")
            self.target_data = target
            self.target_data_size = self.target_data.shape[0]
            print("Target data loaded.")
        print("Data load complete.")

    def loadWeight(self, source_weight=None, target_weight=None):
        if source_weight:
            if source_weight.ndim != 1:
                raise Exception("Source weight must be 1 dimension.")
            if source_weight.shape[0] != self.source_data_size:
                raise Exception("Size mismatch check dimensions of Source weight and data points.")
            self.source_weight = source_weight
            print("Source weights loaded.")
        if target_weight:
            if target_weight.ndim != 1:
                raise Exception("Source weight must be 1 dimension.")
            if target_weight.shape[0] != self.target_data_size:
                raise Exception("Size mismatch check dimensions of Source weight and data points.")
            self.target_weight = target_weight
            print("Target weights loaded.")
        print("Weights load complete.")
            

    def loadDataFromFile(self, source_file=None, target_file=None):
        source_data = None
        target_data = None
        source_weight = None
        target_weight = None
        if source_file:
            source = pd.read_csv(source_file)
            if 'x' in source.columns.tolist() and 'y' in source.columns.tolist():
                source_data = np.array(source['x', 'y'].values())
            else:
                raise Exception("Please ensure that 'x' and 'y' are columns in source file.")
            if 'w' in source.columns.tolist():
                source_weight = np.array(source['w'])
            else:
                print("Warning: No Source weight specified if it was in file please set column to 'w'.")
        if target_file:
            target = pd.read_csv(target_file)
            if 'x' in target.columns.tolist() and 'y' in target.columns.tolist():
                target_data = np.array(target['x', 'y'].values())
            else:
                raise Exception("Please ensure that 'x' and 'y' are columns in source file.")
            if 'w' in target.columns.tolist():
                target_weight = np.array(target['w'])
            else:
                print("Warning: No Source weight specified if it was in file please set column to 'w'.")
        self.loadData(source_data, target_data)
        self.loadWeight(source_weight, target_weight)

    def makeTransportPlan(self):
        if self.source_data and self.target_data:
            if self.source_data_size == self.target_data_size:
                loss_matrix = ot.dist(self.source_data, self.target_data)
                loss_matrix = loss_matrix / loss_matrix.max()
                if not self.source_weight:
                    self.source_weight = np.ones((self.source_data_size,)) / self.source_data_size
                    print("The Source weights are intiialized to one. If custome weight please load weight.")
                if not self.target_weight:
                    self.target_weight = np.ones((self.target_data_size,)) / self.target_data_size
                    print("The Target weights are intiialized to one. If custome weight please load weight.")
                transport = ot.emd(self.source_weight, self.target_weight, loss_matrix, log = True)
                print("Transport Plan Complete")
                print("The cost is: {}".format(transport[1]['cost']))
                self.transport_plan = transport[0]
                return transport
            else:
                print("Optimal Transport Plan not complete due to mismatch in Source and Target Size.")
                return
        else:
            print("Optimal Transport Plan not complete. Please add Source & Target data and rerun.")
            return



