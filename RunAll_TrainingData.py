# Import modules
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
# My python scripts
from ImportData import *
from TrainTestGPModel import *

from remove_ind import *

datadir = GetDir()
time = 10

filename = datadir+'Inputs_0-{}_yrs_Outputs_70-100_yrs.nc'.format(time)
(X_SfcTemp,X_AirTemp500,X_GeoHeight500,X_SLP,X_RF,y,
                lons,lats,lons1,lats1,Names) = OpenFile(filename)
Noisy = ['No OC Global', 'No Dust Arabia','No VOC Global','No SO2 India','No NOX Global','No BC Europe','No BC East Asia','No BC US']
for runname in Noisy:
       (Names,[X_SfcTemp,X_GeoHeight500,X_RF,y]) = remove_ind(Names,runname,[X_SfcTemp,X_GeoHeight500,X_RF,y] )


# Predictor variable
X = X_SfcTemp.copy()

(N,p) = X.shape

area = Area(lons1,lats1)
area_flat = area.flatten()
print(area_flat.shape,X_SfcTemp.shape)
regions = ['Europe','Africa','US','South_America','East_Asia','India']

# Save to
plot_dir = '/rds/general/user/lm2612/home/Outputs/TrainingData/Inputs_0-{}_yrs_Outputs_70-100_yrs_'.format(time)

# Test data set
print(Names)

random.seed(123)
# Subset multiple times
Ntrain = int(sys.argv[1])

Ntest = N - Ntrain 
NamesForNtest = []

for i in range(200):
    inds = random.sample(range(N), Ntest)
    names = [Names[ind] for ind in inds]
    NamesForNtest.append(names)

if __name__ == '__main__':
    # Define an output queue
    output = mp.Queue()

    # Loop over all Names as testdata
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=TrainTestFull, args=(X,y,Names,names,lons,lats,lons1,lats1,
         area_flat,plot_dir+str(Ntrain)+'_run_'+str(runindex)+'_')) for (names,runindex) in zip(NamesForNtest,range(0,len(NamesForNtest)))]
 
    # Run processes
    for p in processes:
        p.start()

        # Exit the completed processes
    for p in processes:
        p.join()

print('done')




