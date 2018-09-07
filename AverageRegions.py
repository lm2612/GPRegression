import numpy as np
from RegionLatitudes import *
from DefineRegions import *
def AverageRegions(X,lons,lats,region_list,area_flat):
    """ Returns X in shape of regions """
    assert ( X.shape[1] == area_flat.shape[0] ),'Dimensions of X should be the same as dimension of area. X dim  = {}, Area dim = {} '.format(X.shape[1],area_flat.shape[0] )
    print(area_flat) 
    X_new = np.zeros((X.shape[0],len(region_list) ))
    for (i,region) in enumerate(region_list):
        grid = DefineRegion(region,lons,lats)
        area_region = grid.flatten()*area_flat
        X_new[:,i] = np.average(X,axis=1,weights=area_region)
    return(X_new)
