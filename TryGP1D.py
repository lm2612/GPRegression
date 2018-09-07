# Import modules
import numpy as np
import matplotlib.pyplot as plt
import GPy 

# My python scripts
from ImportData import *
from SplitTrainingAndTesting import split_set,split_set_random


# Predictor variable
X = X_SfcTemp.copy()

# Try predicting means
# Need area as well
Xmean = np.average(X,axis=1)[:,None]
ymean = np.average(y,axis=1)[:,None]
(N,p) = X.shape
e = int(p/80)
print(e)
X = X[:,20000:20001]
print(Xmean.shape)
y = y[:,20000:20001]
# Test data set
print(Names)

for i in range(1):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set_random(X,y,Names,5,20*i**2-i*4+3)

    (N,p) = Xmean.shape
    kern = GPy.kern.Matern52(p)+GPy.kern.Linear(p)
    m = GPy.models.GPRegression(X_train,y_train,kern)
    m.optimize()

    # Test
    y_pred,cov = m.predict_noiseless(X_test,full_cov=True)
    print(' pred,        test ')
    for j in range(y_test.shape[0]):
         print('{:<3f} {:>3f}'.format(y_pred[j,0],y_test[j,0]))
    m.plot(visible_dims=[0])
    plt.plot(X_test[:,0],y_test[:,0],'r^')

    plt.show()
