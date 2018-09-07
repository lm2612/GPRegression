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

# Test data set
print(Names)
TestName = ['No CO Global','No SO2 US','3X CH4']

#(X_train,X_test,y_train,y_test,names_train,names_test) = split_set(Xmean,ymean,Names,TestName)
for i in range(5):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set_random(Xmean,ymean,Names,5,20*i**2-i*4+3)


    print(y_test.shape)
    kern = GPy.kern.RBF(1) + GPy.kern.Linear(1)
    m = GPy.models.GPRegression(X_train,y_train,kern)
    m.optimize()

    # Test
    y_pred,cov = m.predict_noiseless(X_test,full_cov=True)

    m.plot()
    plt.plot(X_test,y_test,'r^')

    plt.show()
