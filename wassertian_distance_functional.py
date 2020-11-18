import sys
import numpy as np
import ot

def getWassertianDistance(source, target):
    loss_matrix = ot.dist(source, target)
    loss_matrix = loss_matrix / loss_matrix.max()
    source_weights = np.ones((source.shape[0],)) / source.shape[0]
    target_weights = np.ones((target.shape[0],)) / target.shape[0]
    transport_map = ot.emd(source_weights, target_weights, loss_matrix, log = True)
    return transport_map[1]['cost']


def getArray(filename, samples, features):
    array = None
    with open(filename, 'r') as f:
        file_string = f.read()
        distrib = [float(i) for i in file_string.split(',')]
        array = np.array(distrib).reshape(samples, features)
    return array

def execution(features, source_filename, source_samples, target_filename, target_samples):
    print("The distance is: {}".format(getWassertianDistance(getArray(source_filename, source_samples, features), getArray(target_filename, target_samples, features))))

if __name__ == "__main__":
    if len(sys.argv) == 6:
        execution(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))