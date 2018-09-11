# Import modules
import numpy as np
import matplotlib.pyplot as plt
import GPy 
from sklearn.decomposition import PCA

# My python scripts
from ImportData import *
from SplitTrainingAndTesting import split_set,split_set_random
from AreaWeighting import Area,Area1
from AverageRegions import AverageRegions
from PlotPredictionsVsTrue_NoScaling import PredictionPlot

# Predictor variable
X = X_SfcTemp.copy()

(N,p) = X.shape
area = Area(lons1,lats1)
area_flat = area.flatten()
print(area_flat.shape,X_SfcTemp.shape)
regions = ['Europe','Africa','US','South_America','East_Asia','India']

# Save to
plot_dir = '/work/lm2612/GPPCA/'

# Test data set
print(Names)


for i in range(1):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set_random(X,y,Names,5,20*i**2-i*4+3)
    X_PCA = PCA().fit(X_train)
    y_PCA = PCA().fit(y_train)

    X_trans = X_PCA.transform(X_train)
    y_trans = y_PCA.transform(y_train)

    (N,p) = X_trans.shape
    print(p)
    kern = GPy.kern.Matern52(p,ARD=True)+GPy.kern.Linear(p,ARD=True)
    
    m = GPy.models.GPRegression(X_trans,y_trans,kern)
    m.optimize()

    
    # Test
    y_test_trans = y_PCA.transform(y_test)
    X_test_trans = X_PCA.transform(X_test)

    y_pred,cov = m.predict_noiseless(X_test_trans,full_cov=True)
    print(' pred,        test ')
    for j in range(y_test_trans.shape[0]):
         print('{:<3f} {:>3f}'.format(y_pred[j,0],y_test_trans[j,0]))

    
    for k in range(p):
        plt.clf()
        m.plot(visible_dims=[k])
        plt.plot(X_test_trans[:,0],y_test_trans[:,0],'r^')

        plt.savefig(plot_dir+'GP_plot_dim_{}.png'.formaat(k))
    

    y_pred_full = y_PCA.inverse_transform(y_pred)
    # check output
    rmse = np.sqrt(  np.average(( y_test - y_pred_full)**2.,axis=1,weights=area_flat )

    PredictionPlot(y_test,y_pred_full,lons,lats, names_train, names_test, plot_dir,rmses)

