import netCDF4
import os
# Get dictionary called filenames containing file names for each file code
from FileCodesToName import *

def OpenFile(filename):
    """ filename is the name of file to open, must be correct directory """
    print("Opening %s ... "%(filename))
    # Copy dimension and information for meta data
    dataset = netCDF4.Dataset(filename, 'r',format='NETCDF4_CLASSIC')

    # Get sample list
    Samples = dataset.variables['samples_list'][:]
    Files = netCDF4.chartostring(Samples)

    # Variables we will extract
    ListOfVarNames = ['SfcTemp','AirTemp','GeoHeight','SLP',
                      'RF','SfcTempResponse']
    ListOfVars = []
        
    # Get variables and add to ListOfVars
    # Throw warning but not error if these dont exist
    for VarName in ListOfVarNames:
        try: 
            Var = dataset.variables[VarName][:]
        except KeyError: 
            print("Warning: Variable {0} does not exist in file {1}."
                " Continue for now but this may cause issues later ".format(VarName,filename))
            Var = None
        ListOfVars.append(Var)

    lons = dataset.variables['longitude'][:]
    lons1 = dataset.variables['longitude_1'][:]
    lats = dataset.variables['latitude'][:]
    lats1 = dataset.variables['latitude_1'][:]

    filecodes = Files
    Names = []
    for i in range(len(filecodes)):
        Names.append(filenames[filecodes[i]])
    nlon = len(lons)
    nlat = len(lats)  
    print("Scenarios: {0}".format(Names))
    print("Done opening file ") 
    
    ReturnVars = ListOfVars + [lons,lats,lons1,lats1,Names]
    return(ReturnVars)

