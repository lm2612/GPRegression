# Import modules
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
       (Names,[X_SfcTemp,X_AirTemp500,X_GeoHeight500,X_SLP,X_RF,y]) = remove_ind(Names,runname,[X_SfcTemp,X_AirTemp500,X_GeoHeight500,X_SLP,X_RF,y] )



CO_ind = Names.index('No CO Global')
X_RF_CO = X_RF[CO_ind]

print(Names)
X = X_RF
CO2_ind =(Names.index('2X CO2'))
CO2_change = y[CO2_ind]
y_coupled = y
Names_coupled = Names

print(CO2_change.shape)
area_flat = Area(lons1,lats1).flatten()
filename=datadir+'All_Atmos_incRF_-1yr_correctedECLIPSE.nc'
(X_RF,lons,lats,lons1,lats1,Names) = OpenFile(filename,ListOfVarNames=['RF_1'])
Noisy = ['No OC Global','No Dust Arabia','No VOC Global','No SO2 India','No NOX Global','No BC Europe','No BC East Asia','No BC US']
for runname in Noisy:
    if runname in Names:
       (Names,[X_RF]) = remove_ind(Names,runname,[X_RF] )

# Overwrite CO summer and winter with RF
COsummer_ind = Names.index('No CO Summer')
COwinter_ind = Names.index('No CO Winter')
print(len(Names))
print(Names,Names_coupled)
N = len(Names)-1
X_RFnew = np.zeros((N, X_RF.shape[1]))

# First overwrite one row (winter) with global 
Names[COwinter_ind] = 'No CO Global'
X_RF[COwinter_ind] = X_RF_CO
# Then remove summer row
Names.remove('No CO Summer')
X_RF=np.delete(X_RF, COsummer_ind, axis=0)

print(X_RF)

X = X_RF

# Need to re-order y so its aligned with X
y_true = np.zeros(y.shape)

for i in range(len(Names)):
    if Names[i] in Names_coupled:
        coupled_index = Names_coupled.index(Names[i])
        y_true[i,:] = y_coupled[coupled_index,:]
    else:
        print(Names[i],'not in coupled')


y = y_true
X = X_RF
area_flat = Area(lons1,lats1).flatten()
# Predictor variable
Xname = 'ERF'
(N,p) = X.shape

regions = ['Europe','Africa','US','South_America','East_Asia','India']

# Save to
plot_dir = '/rds/general/user/lm2612/home/Outputs/Predictors/Outputs_70-100_{}_'.format(Xname)


# Test data set
print(Names)

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




