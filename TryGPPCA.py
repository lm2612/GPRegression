# Import modules
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import GPy 
from sklearn.decomposition import PCA
import pickle

# My python scripts
from ImportData import *
from SplitTrainingAndTesting import split_set,split_set_random
from AreaWeighting import Area,Area1
from AverageRegions import AverageRegions
from PlotPredictionVsTrue_NoScaling import PredictionPlot
from Metrics import *
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


for i in range(len(Names)):
    print("Training Stage:")
    TestName = Names[i]

    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set(X,y,Names,TestName)
    # PCA
    X_PCA = PCA().fit(X_train)
    y_PCA = PCA().fit(y_train)
    X_trans = X_PCA.transform(X_train)
    y_trans = y_PCA.transform(y_train)

    # GP Model
    (N,p) = X_trans.shape
    kern = GPy.kern.Matern52(p,ARD=True)+GPy.kern.Linear(p,ARD=True)    

    print("kernel used: {}".format(kern))

    m = GPy.models.GPRegression(X_trans,y_trans,kern)
    m.optimize()

    print("Model optimized: {}".format(m))

    # Validation
    print("Validation Stage: Predicting and plotting... ")
    y_test_trans = y_PCA.transform(y_test)
    X_test_trans = X_PCA.transform(X_test)

    y_pred,cov = m.predict_noiseless(X_test_trans,full_cov=True)
    print(' pred,        test ')
    for j in range(y_test_trans.shape[0]):
         print('{:<3f} {:>3f}'.format(y_pred[j,0],y_test_trans[j,0]))

    # plot
    for k in range(p):
        m.plot(visible_dims=[k])
        plt.plot(X_test_trans[:,0],y_test_trans[:,0],'r^')

        plt.savefig(plot_dir+'GP_plot_dim_{}.png'.format(k))
        plt.close() 

    # invert PC
    y_pred_full = y_PCA.inverse_transform(y_pred)

    rmses = np.sqrt(  np.average(( y_test - y_pred_full)**2.,axis=1,weights=area_flat ))
    PredictionPlot(y_test,y_pred_full,lons,lats, names_train, names_test, plot_dir,rmses)
    
    # compare RMSEs for key regions
    metric_regions = regions+['Global']
    y_test_regions = AverageRegions(y_test,lons,lats,metric_regions,area_flat)
    y_pred_regions = AverageRegions(y_pred_full,lons,lats,metric_regions,area_flat)
    RegionalMetrics(y_pred_regions,y_test_regions,metric_regions,lons1,lats1,names_train,names_test,plot_dir,area_flat)
    
    # save output
    filename = (plot_dir + 'output')
    print("Completed. Saving data as %s ..."%filename)
    output = {'names_train':names_train, 'names_test':names_test,
              'X_train':X_train, 'X_test':X_test,
              'y_train':y_train, 'y_test':y_test, 'y_pred':y_pred_full,
              'lons':lons,'lats':lats,'lons1':lons1,'lats1':lats1,
              'area_flat':area_flat, 'GPmodel':m , 'kernel':kern }
    pickle.dump(output,open(filename,'wb') )
    print("Data saved")
