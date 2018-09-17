# Import modules
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing as mp

# My python scripts
from ImportData import *
from TrainTestGPModel import *

# Predictor variable
X = X_SfcTemp.copy()

(N,p) = X.shape
area = Area(lons1,lats1)
area_flat = area.flatten()
print(area_flat.shape,X_SfcTemp.shape)
regions = ['Europe','Africa','US','South_America','East_Asia','India']

# Save to
n_comp = 1
plot_dir = ('/home/laura/Documents/PCA/pca_out_{}'.format(n_comp))

# Test data set
print(Names)

# our function of interest for PCA is
if __name__ == '__main__':
    # Define an output queue
    output = mp.Queue()

    # Loop over all Names as testdata
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=TrainTestPCA, args=(X,y,Names,[name],lons,lats,lons1,lats1,area_flat,
        plot_dir,n_comp,False,True)) for name in Names]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    print('done')




