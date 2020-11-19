from PlotPredictionVsTrue_NoScaling import *
from plotmapfunction import *
import numpy as np
import pandas as pd
#from ImportData import *
import GPy
# Try diff lengthscales (degrees)
Lengthscales = [1.,5.,10.,30.,45.,60.,90.,100.,180.,360.,1000.]
p = 2  # lon and lat
nlon,nlat = 192,145
#nlon,nlat = 60,30
lats = np.linspace(-90.,90.,nlat)
lons = np.linspace(0.,360.,nlon)
# our inputs are lats and lons
longrid = np.zeros((nlat,nlon))
latgrid = np.zeros((nlat,nlon))

for i in range(nlat):
    longrid[i,:] = lons

for j in range(nlon):
    latgrid[:,j] = lats

N = nlon*nlat
X = np.array([latgrid.flatten(),longrid.flatten()]).T
print(X)
y = np.zeros((N))[:,None]

kern = GPy.kern.RBF(p,ARD=True,lengthscale = 5*np.ones(p))#+GPy.kern.Linear(p,ARD=True)
m = GPy.models.GPRegression(X,y,kern)
m.optimize()
print(m.kern.lengthscale)
Lengthscales = [5.]
for L in Lengthscales:
    print(L)
    lengths = L* np.ones(p)
    kern = GPy.kern.RBF(p,ARD=True,lengthscale=lengths)#+GPy.kern.Linear(p,ARD=True)
    m = GPy.models.GPRegression(X,y,kern)

    m.optimize()
    #m.randomize()
    print(m)
    kern = m.kern
    print(kern.lengthscale)
    # Draw some random samples from prior

    mu = np.zeros((N))
    C = kern.K(X,X)

    plt.clf()
    plt.figure()
    y_pred = np.zeros(N)
    Z  = np.random.multivariate_normal(mu,C, 1)
    print('Z',Z)
    print(Z.shape)
    y_pred = np.reshape(Z,(nlat,nlon))
    #for i in range(nlat):
    #    for j in range(nlon):
    #        y_pred[i,j] = Z[i]    
    #y_pred = np.reshape(y_pred[:,:],((len(lats),len(lons))))
    save = '/home/lm2612/plotGP/drawrandom/prior_map_lengthscale{}.png'.format('optimized10,11')
    max_val = (0.7*np.max(np.abs(y_pred)))
    print(max_val)
    levels = np.linspace(-1.0*max_val,max_val,50) 
    print(levels)

    plotmap(lons,lats,y_pred,savefile=save, cmap="RdBu_r", levels=levels,
            variable_label='',plottitle='Prior',plotaxis=None,colorbar=1.0)

    
    y_pred = y_pred.flatten()
    y_true = y[0,:]

    y_mean = np.mean(y)
    y_var = np.var(y)
    y_rescaled = ((y_pred-np.mean(y_pred))/np.var(y_pred)+y_mean)*y_var
    rmse = np.sqrt( np.mean( (y_rescaled - y_true )**2. ) )
    print('Lengthscale: ',L)
    print('RMSE: ',rmse)

    data = {'y_pred':y_pred}
    df = pd.DataFrame(data,columns=['y_pred'])
    df.to_csv('/work/lm2612/GPprior/prediction.csv')

