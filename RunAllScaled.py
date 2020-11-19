# Import modules
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing as mp

# My python scripts
from GetDir import *
from ReadAvgFiletyr import *
from TrainTestGPModel import *

from remove_ind import *

datadir = GetDir()
time = 10
end_time = 100
filename = datadir+'Inputs_0-{}_yrs_Outputs_70-{}_yrs.nc'.format(time,end_time)
(X_SfcTemp,X_AirTemp500,X_GeoHeight500,X_SLP,X_RF,y,
                lons,lats,lons1,lats1,Names) = OpenFile(filename)
Noisy = ['No BB Tropics','10X BB Tropics','No BB Africa', 'No OC Global', 'No Dust Arabia','No VOC Global','No SO2 India','No NOX Global','No BC Europe','No BC East Asia','No BC US']
for runname in Noisy:
    if runname in Names:
       (Names,[X_SfcTemp,X_GeoHeight500,X_RF,y]) = remove_ind(Names,runname,[X_SfcTemp,X_GeoHeight500,X_RF,y] )

# Predictor variable
X = X_SfcTemp.copy()

(N,p) = X.shape

area = Area(lons1,lats1)
area_flat = area.flatten()
print(area_flat.shape,X_SfcTemp.shape)
regions = ['Europe','Africa','US','South_America','East_Asia','India']

# Save to
plot_dir = '/rds/general/user/lm2612/home/Outputs/GP/ScaledInputs_0-{}_yrs_ScaledOutputs_70-{}_yrs'.format(time,end_time)

# Test data set
print(Names)

# our function of interest for PCA is
if __name__ == '__main__':
    # Define an output queue
    output = mp.Queue()

    # Loop over all Names as testdata
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=TrainTestScale, args=(X,y,Names,[name],lons,lats,lons1,lats1,area_flat,plot_dir)) for name in Names]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    print('done')




