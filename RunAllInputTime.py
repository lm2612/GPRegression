# Import modules
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing as mp

# My python scripts
import numpy as np
from GetDir import *
from ReadAvgFiletyr import *
from TrainTestGPModel import *

from remove_ind import *

datadir = GetDir()
time = sys.argv[1]
print("Running with {} input years".format(time))

filename = datadir+'Inputs_0-{}_yrs_Outputs_70-100_yrs.nc'.format(time)
print("Input file =",filename)
(X_SfcTemp,X_AirTemp500,X_GeoHeight500,X_SLP,X_RF,y,
                lons,lats,lons1,lats1,Names) = OpenFile(filename)
# Remove runs we have calcualted to be to weak and noisy
Noisy = ['No OC Global', 'No Dust Arabia','No VOC Global','No SO2 India','No NOX Global','No BC Europe','No BC East Asia','No BC US']
for runname in Noisy:
       (Names,[X_SfcTemp,X_GeoHeight500,X_RF,y]) = remove_ind(Names,runname,[X_SfcTemp,X_GeoHeight500,X_RF,y] )


# Predictor variable
X = X_SfcTemp.copy()

(N,p) = X.shape
# Area
area = Area(lons1,lats1)
area_flat = area.flatten()
print(area_flat.shape,X_SfcTemp.shape)

# Save to
plot_dir = '/rds/general/user/lm2612/home/Outputs/Inputs_Nyear/Inputs_0-{}yrs_Outputs_70-100_yrs_GP_'.format(time)

# Test data set
print(Names)

# our function of interest for PCA is
if __name__ == '__main__':
    # Define an output queue
    output = mp.Queue()

    # Loop over all Names as testdata
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=TrainTestFull, args=(X,y,Names,[name],lons,lats,lons1,lats1,area_flat,plot_dir)) for name in Names]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    print('done')




