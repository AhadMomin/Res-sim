import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from OneDim import OneDimReservoir
from TwoDim import TwoDimReservoir


class Project2(TwoDimReservoir):
    def __init__(self, inputs):

        super().__init__(inputs)

        return

    def check_input_and_return_data(self, input_name):

        dtype = type(input_name)

        if dtype == str:
            filename = input_name
            data = np.loadtxt(filename)

        elif dtype == list or dtype == tuple:
            data = np.array(input_name)

        elif dtype == int or dtype == float:
            ngrids = self.ngrids
            data = input_name * np.ones(ngrids)

        else:
            raise ValueError("Error - incorrect data type: %s" % (input_name))

        return data
