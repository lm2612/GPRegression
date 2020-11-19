# Import modules
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing as mp

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

# What type of PCA?
arg_number = int(sys.argv[1])
if arg_number == 1:
    pca_on = 'X'
    pca_X = True
    pca_Y = False
elif arg_number == 2:
    pca_on = 'Y'
    pca_X = False
    pca_Y = True
elif arg_number == 3:
    pca_on = 'XY'
    pca_X = True
    pca_Y = False
elif arg_number == 4:
    pca_on = 'None'
    pca_X = False
    pca_Y = False
else:
    print('Argument entered is ',arg_number)
    print('Please enter 1,2,3 or 4 for PCA on X, Y, X and Y or neither')

(N,p) = X.shape
area = Area(lons1,lats1)
area_flat = area.flatten()
print(area_flat.shape,X_SfcTemp.shape)
regions = ['Europe','Africa','US','South_America','East_Asia','India']

# Save to
n_comp = 20
plot_dir = '/rds/general/user/lm2612/home/Outputs/DimRed/Outputs_70-100_yrs_GP_PCA_{}_'.format(pca_on)


# Test data set
print(Names)

# our function of interest for PCA is
if __name__ == '__main__':
    # Define an output queue
    output = mp.Queue()

    # Loop over all Names as testdata
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=TrainTestPCA, args=(X,y,Names,[name],lons,lats,lons1,lats1,area_flat,
        plot_dir,None,pca_X,pca_Y)) for name in Names]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    print('done')




