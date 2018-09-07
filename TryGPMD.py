# Import modules
import numpy as np
import matplotlib.pyplot as plt
import GPy 

# My python scripts
from ImportData import *
from SplitTrainingAndTesting import split_set,split_set_random
from AreaWeighting import Area,Area1
from AverageRegions import AverageRegions

# Predictor variable
X = X_SfcTemp.copy()

(N,p) = X.shape
area = Area(lons1,lats1)
area_flat = area.flatten()
print(area_flat.shape,X_SfcTemp.shape)
regions = ['Europe','Africa','US','South_America','East_Asia','India']
X = AverageRegions(X_SfcTemp,lons,lats,regions,area_flat)
y = AverageRegions(y,lons,lats,regions,area_flat)
X = X[:,:]
y = y[:,0:1]
# Test data set
print(Names)

for i in range(1):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set_random(X,y,Names,5,20*i**2-i*4+3)

    (N,p) = X.shape
    print(p)
    kern = GPy.kern.Matern52(p,ARD=True)+GPy.kern.Linear(p,ARD=True)
    
    m = GPy.models.GPRegression(X_train,y_train,kern)
    m.optimize()

    # Test
    y_pred,cov = m.predict_noiseless(X_test,full_cov=True)
    print(' pred,        test ')
    for j in range(y_test.shape[0]):
         print('{:<3f} {:>3f}'.format(y_pred[j,0],y_test[j,0]))

    for k in range(p):
        m.plot(visible_dims=[k])
        plt.plot(X_test[:,0],y_test[:,0],'r^')

        plt.show()


