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
from GPerrors import *
from RegionLatitudes import *
from PlotGPErrs import *

def TrainModel(X_train,y_train,X_test,y_test):
    print("Training Model...")
    # GP Model
    (N,p) = X_train.shape
    y_pred = np.zeros(y_test.shape)
    cov = np.zeros(y_test.shape)
    for i in range(y_train.shape[1]):
        sigma = np.std(y_train[:,i])
        variance = sigma**2.
        kern = GPy.kern.RBF(p,variance=variance,ARD=True)+GPy.kern.Linear(p,ARD=True)
        m = GPy.models.GPRegression(X_train,y_train[:,i].reshape(-1,1),kern)
        y_pred_i,cov_i = m.predict(X_test,full_cov=False)
        y_pred[:,i] = y_pred_i[:,0]
        cov[:,i] = cov_i[:,0]

        if i%1000==0:
            print("Iteration ",i)
            print("Sigma ",sigma)

    return(y_pred,cov)

def TestModel(m,X_test,y_test):
    # Validation
    print("Validation Stage: Predicting and plotting... ")
    y_pred = np.zeros(y_test.shape)
    cov = np.zeros(y_test.shape)
    for i in range(len(m)):
        m_i = m[i]
        y_pred_i,cov_i = m_i.predict(X_test,full_cov=False)
        y_pred[:,i] = y_pred_i[:,0]
        cov[:,i] = cov_i[:,0]
        
    print(' pred,        test ')
    for j in range(y_test.shape[0]):
         print('{:<3f} {:>3f}'.format(y_pred[j,0],y_test[j,0]))
    print(cov)
    """
    print(X_test.shape)
    print(y_test.shape)
    print(cov)
    print(cov.shape)
    err = m.predict_quantiles(X_test,quantiles=[2.5])
    print('2.5 quantile err')
    print(err[0].shape)
    print(err[0])
    gp_err = y_pred- (err[0])i
    print(gp_err.shape)
    print(gp_err)"""
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
    
def Validation(y_test,y_pred,y_cov,lons,lats,lons1,lats1,names_train,names_test,plot_dir,area_flat,metric_regions=['Global']):
    # compare RMSEs for key regions
    y_test_regions = AverageRegions(y_test,lons,lats,metric_regions,area_flat)
    y_pred_regions = AverageRegions(y_pred,lons,lats,metric_regions,area_flat)
    RegionalMetrics(y_pred,y_test,metric_regions,lons,lats,lons1,lats1,names_train,names_test,plot_dir,area_flat)
    GPError(y_cov,y_pred,y_test,metric_regions,lons,lats,lons1,lats1,names_train,names_test,plot_dir,area_flat)

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
    # Train and pred
    y_pred,cov  = TrainModel(X_train,y_train,X_test,y_test)
    # Plot
    PlotMap(y_test,y_pred,lons,lats,names_train,names_test,plot_dir,area_flat)
    PlotErrs(cov,lons,lats,names_train,names_test,plot_dir,area_flat)
    regions_all = ['Global']+list(RegionsList)
    print(y_pred.shape)
    
    for v in range(len(TestName)):
        Validation(y_test[v:v+1,],y_pred[v:v+1,:],cov[v:v+1,:],lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[v]+'_',area_flat,metric_regions=regions_all)
    #Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir+TestName[0]+'_')

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
        cov = y_PCA.inverse_transform(cov)
    else:
        y_pred_full = y_pred
    # Plot
    regions_all = ['Global']+list(RegionsList)

    PlotMap(y_test,y_pred_full,lons,lats,names_train,names_test,plot_dir,area_flat)
    Validation(y_test,y_pred_full,cov,lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[0]+'_',area_flat,metric_regions=regions_all)
    Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred_full,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir+TestName[0]+'_')

