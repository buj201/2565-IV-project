import rpy2.robjects as robjects
import numpy as np

def to_array(x):
    '''
    Convert r vector to numpy array
    '''
    return np.array(list(x))

def to_FloatVector(x):
    return robjects.FloatVector(list(x.flatten()))