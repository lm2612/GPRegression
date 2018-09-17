# Import modules
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import GPy 
from sklearn.decomposition import PCA
import pickle

# My python scripts
from SplitTrainingAndTesting import split_set,split_set_random
from AverageRegions import AverageRegions
from PlotPredictionVsTrue_NoScaling import PredictionPlot
from Metrics import *


def TrainModel(X_train,y_train):
    print("Training Model...")
    # GP Model
    (N,p) = X_train.shape
    kern = GPy.kern.RBF(p,ARD=True)+GPy.kern.Linear(p,ARD=True)    

    print("kernel used: {}".format(kern))

    m = GPy.models.GPRegression(X_train,y_train,kern)
    m.optimize()

    print("Model optimized: {}".format(m))
    return(m,kern)

def TestModel(m,X_test,y_test):
    # Validation
    print("Validation Stage: Predicting and plotting... ")
    y_pred,cov = m.predict_noiseless(X_test,full_cov=True)
    print(' pred,        test ')
    for j in range(y_test.shape[0]):
         print('{:<3f} {:>3f}'.format(y_pred[j,0],y_test[j,0]))
    return(y_pred,cov)

def PlotGP(m,X_test,y_test,plot_dir):
    (N,p) = X_test.shape
    for k in range(p):
        m.plot(visible_dims=[k])
        plt.plot(X_test[:,0],y_test[:,0],'r^')

        plt.savefig(plot_dir+'GP_plot_dim_{}.png'.format(k))
        plt.close() 

def PlotMap(y_test,y_pred,lons,lats,names_train,names_test,plot_dir,area_flat):
    rmses = np.sqrt(  np.average(( y_test - y_pred)**2.,axis=1,weights=area_flat ))
    PredictionPlot(y_test,y_pred,lons,lats, names_train, names_test, plot_dir,rmses)
    
def Validation(y_test,y_pred,lons,lats,lons1,lats1,names_train,names_test,plot_dir,area_flat,metric_regions=['Global']):
    # compare RMSEs for key regions
    y_test_regions = AverageRegions(y_test,lons,lats,metric_regions,area_flat)
    y_pred_regions = AverageRegions(y_pred,lons,lats,metric_regions,area_flat)
    RegionalMetrics(y_pred_regions,y_test_regions,metric_regions,lons1,lats1,names_train,names_test,plot_dir,area_flat)

def Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir):
    # save output
    filename = (plot_dir + 'output')
    print("Completed. Saving data as %s ..."%filename)
    output = {'names_train':names_train, 
              'names_test':names_test,
              'X_train':X_train, 
              'X_test':X_test,
              'y_train':y_train,  
              'y_test':y_test,  
              'y_pred':y_pred,
              'lons':lons,
              'lats':lats,
              'lons1':lons1, 
              'lats1':lats1,
              'area_flat':area_flat, 
              'GPmodel':m , 
              'kernel':kern }
    pickle.dump(output,open(filename,'wb') )
    print("Data saved")

def TrainTestFull(X,y,Names,TestName,lons,lats,lons1,lats1,area_flat,plot_dir):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set(X,y,Names,TestName)
    # Train
    (m,kern)  = TrainModel(X_train,y_train)
    # Predict
    y_pred,cov = TestModel(m,X_test,y_test)
    # Plot
    PlotMap(y_test,y_pred,lons,lats,names_train,names_test,plot_dir,area_flat)
    Validation(y_test,y_pred,lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[0]+'_',area_flat,metric_regions=['Global'])
    Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir+TestName[0]+'_')

def TrainTestPCA(X,y,Names,TestName,lons,lats,lons1,lats1,area_flat,plot_dir,n_comp=None,pca_inputs = True,pca_outputs=True):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set(X,y,Names,TestName)
    # PCA
    if pca_inputs is True:
        X_PCA = PCA(n_comp).fit(X_train)
        X_trans = X_PCA.transform(X_train)
        X_test_trans = X_PCA.transform(X_test)
    else:
        X_trans = X_train
        X_test_trans = X_test
    if pca_outputs is True:
        y_PCA = PCA(n_comp).fit(y_train)
        y_trans = y_PCA.transform(y_train)
        y_test_trans = y_PCA.transform(y_test)
    else:
        y_trans = y_train
        y_test_trans = y_test
    # Train
    (m,kern)  = TrainModel(X_trans,y_trans)
    y_pred,cov = TestModel(m,X_test_trans,y_test_trans)
    # Inverse PC
    if pca_outputs is True:
        y_pred_full = y_PCA.inverse_transform(y_pred)
    else:
        y_pred_full = y_pred
    # Plot

    PlotMap(y_test,y_pred_full,lons,lats,names_train,names_test,plot_dir,area_flat)
    Validation(y_test,y_pred_full,lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[0]+'_',area_flat,metric_regions=['Global'])
    Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred_full,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir+TestName[0]+'_')

