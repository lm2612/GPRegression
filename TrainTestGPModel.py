# Import modules
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import GPy 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler as StandardScaler
import pickle

# My python scripts
from AreaWeighting import *
from SplitTrainingAndTesting import split_set,split_set_random
from AverageRegions import AverageRegions
from PlotPredictionVsTrue_NoScaling import PredictionPlot
from Metrics import *
from SlidingWindowRMSE import *
from GPerrors import *
from RegionLatitudes import *

def TrainModel(X_train,y_train):
    print("Training Model...")
    # GP Model
    (N,p) = X_train.shape
    #kern = GPy.kern.RBF(p,ARD=True)**GPy.kern.Coregionalize(1,output_dim=y_train.shape[1],active_dims=[2], rank=4)+GPy.kern.Linear(p,ARD=True)**GPy.kern.Coregionalize(1,output_dim=y_train.shape[1], active_dims=[2],rank=4)
    print(N,p)
    kern = GPy.kern.RBF(p,ARD=True)+GPy.kern.Linear(p,ARD=True)

    print("kernel used: {}".format(kern))
    #icm = GPy.util.multioutput.ICM(input_dim=p,num_outputs=p,kernel=kern)
    #m = GPy.models.GPCoregionalizedRegression(X_train,y_train,icm)
    m = GPy.models.GPRegression(X_train,y_train,kern)
    m.optimize()

    print("Model optimized: {}".format(m))
    return(m,kern)

def TestModel(m,X_test,y_test):
    # Validation
    print("Validation Stage: Predicting and plotting... ")
    print(np.mean(X_test))
    y_pred,cov = m.predict(X_test,full_cov=True)
    print(' pred,        test ')
    print(np.mean(y_pred), np.mean(y_test))
    #for j in range(y_test.shape[0]):
    #     print('{:<3f} {:>3f}'.format(y_pred[j,0],y_test[j,0]))
    print(X_test.shape)
    print(y_test.shape)
    print(cov)
    print(cov.shape)
    err = m.predict_quantiles(X_test,quantiles=[2.5])
    print('2.5 quantile err')
    print(err[0].shape)
    print(err[0])
    gp_err = y_pred- (err[0])
    print(gp_err.shape)
    print(gp_err)
    cov = gp_err
    return(y_pred,cov)


def PlotGP(m,X_test,y_test,plot_dir):
    (N,p) = X_test.shape
    for k in range(p):
        m.plot(visible_dims=[k])
        plt.plot(X_test[:,0],y_test[:,0],'r^')

        plt.savefig(plot_dir+'GP_plot_dim_{}.png'.format(k))
        plt.close() 

def PlotMap(y_test,y_pred,lons,lats,names_train,names_test,plot_dir,area_flat):
    print(y_test.shape, y_pred.shape, area_flat.shape)
    rmses = np.sqrt(  np.average(( y_test - y_pred)**2.,axis=1,weights=area_flat ))
    PredictionPlot(y_test,y_pred,lons,lats, names_train, names_test, plot_dir,rmses)
    
def Validation(y_test,y_pred,y_cov,lons,lats,lons1,lats1,names_train,names_test,plot_dir,area_flat,metric_regions=['Global']):
    # compare RMSEs for key regions
    y_test_regions = AverageRegions(y_test,lons,lats,metric_regions,area_flat)
    y_pred_regions = AverageRegions(y_pred,lons,lats,metric_regions,area_flat)
    RegionalMetrics(y_pred,y_test,metric_regions,lons,lats,lons1,lats1,names_train,names_test,plot_dir,area_flat)
    slidingwindowSizes = [(1,1), (3,3), (5,5), (5,9), (9,5), (9,9), (15,15), (25,25), (45,45), (45,25), (25,45)]
    SlidingWindowMetrics(y_pred,y_test,slidingwindowSizes,lons,lats,lons1,lats1,names_train,names_test,plot_dir,area_flat)
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
    # Train
    (m,kern)  = TrainModel(X_train,y_train)
    # Predict
    y_pred,cov = TestModel(m,X_test,y_test)
    # Plot
    PlotMap(y_test,y_pred,lons,lats,names_train,names_test,plot_dir,area_flat)
    regions_all = ['Global']+list(RegionsList)
    print(y_pred.shape)
    for v in range(len(TestName)):
        Validation(y_test[v:v+1,],y_pred[v:v+1,:],cov[v:v+1,:],lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[v]+'_',area_flat,metric_regions=regions_all)
    Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir+TestName[0]+'_')

def TrainTestScale(X,y,Names,TestName,lons,lats,lons1,lats1,area_flat,plot_dir):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set(X,y,Names,TestName)
    # Scale
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    X_train_norm  = scaler_x.transform(X_train)
    X_test_norm = scaler_x.transform(X_test)
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    y_train_norm  = scaler_y.transform(y_train)
    y_test_norm = scaler_y.transform(y_test)  
    # Train
    (m,kern)  = TrainModel(X_train_norm,y_train_norm)
    # Predict
    y_pred_norm,cov = TestModel(m,X_test_norm,y_test_norm)
    y_pred = scaler_y.inverse_transform(y_pred_norm)
    # Plot
    PlotMap(y_test,y_pred,lons,lats,names_train,names_test,plot_dir,area_flat)
    regions_all = ['Global']+list(RegionsList)
    print(y_pred.shape)
    for v in range(len(TestName)):
        Validation(y_test[v:v+1,],y_pred[v:v+1,:],cov[v:v+1,:],lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[v]+'_',area_flat,metric_regions=regions_all)
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
        cov = y_PCA.inverse_transform(cov)
    else:
        y_pred_full = y_pred
    # Plot
    regions_all = ['Global']+list(RegionsList)

    PlotMap(y_test,y_pred_full,lons,lats,names_train,names_test,plot_dir,area_flat)
    Validation(y_test,y_pred_full,cov,lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[0]+'_',area_flat,metric_regions=regions_all)
    Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred_full,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir+TestName[0]+'_')


def TrainTestRegional(X,y,Names,TestName,lons,lats,lons1,lats1,area_flat,plot_dir,regions=RegionOnlyList,reg_inputs = True,reg_outputs=True):
    (X_train,X_test,y_train,y_test,names_train,names_test) = split_set(X,y,Names,TestName)
    # PCA
    if reg_inputs is True:
        print(regions)
        X_trans = split_into_regions(X_train, Area(lons1,lats1).flatten(),
                      lons, lats, lons1, lats1,
                      regions)
        X_test_trans = split_into_regions(X_test, Area(lons1,lats1).flatten(),
                      lons, lats, lons1, lats1,
                      regions)
    else:
        X_trans = X_train
        X_test_trans = X_test
    if reg_outputs is True:
        print(regions)
        y_trans = split_into_regions(y_train, Area(lons1,lats1).flatten(),
                      lons, lats, lons1, lats1,
                      regions)
        y_test_trans = split_into_regions(y_test, Area(lons1,lats1).flatten(),
                      lons, lats, lons1, lats1,
                      regions)
    else:
        y_trans = y_train
        y_test_trans = y_test

    # Train
    (m,kern)  = TrainModel(X_trans,y_trans)
    y_pred,cov = TestModel(m,X_test_trans,y_test_trans)

    # Inverse PC
    if reg_outputs is True:
        areas = np.ones(len(regions))
        RegionalMetrics(y_pred,y_test_trans,regions,lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[0]+'_',areas)
        GPError(cov,y_pred,y_test_trans,regions,lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[0]+'_',areas)
    else:
        y_pred_full = y_pred
        # Plot
        regions_all = ['Global']+list(RegionsList)
        print(TestName[0])
        PlotMap(y_test,y_pred_full,lons,lats,names_train,names_test,plot_dir,area_flat)
        Validation(y_test,y_pred_full,cov,lons,lats,lons1,lats1,names_train,names_test,plot_dir+TestName[0]+'_',area_flat,metric_regions=regions_all)
    Save(names_train,names_test,X_train,X_test,y_train,y_test,y_pred,lons,lats,lons1,lats1,area_flat,m,kern,plot_dir+TestName[0]+'_')


