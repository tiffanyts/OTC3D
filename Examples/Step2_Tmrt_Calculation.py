# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:23:35 2017
@author: Tiffany Sin, Negin Nazarian 2017

TMRT for idealized Matrices. This example is dependent on Step1_Model_SetUp. 

"""
#import pyliburo
#import ExtraFunctions
import datetime
import time
from datetime import datetime, date, time
import thermalcomfort 

# if there are problems with importing thermal comfort, you can directly input the source code from the pop up window 
# by using the following 3 lines. 
#import easygui
#print("Navigate to thermal comfort file ")
#thermalcomfortpath=easygui.fileopenbox()

# IN CASE PYLIBURO NEEDS TO BE REINSTALLED 
def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
        print "Checking for package.."
    except ImportError:
        import pip
        pip.main(['install', package])
        print "Package not available. Importing package.."
    finally:
        globals()[package] = importlib.import_module(package)
        print "Package installed"

# PATHS         
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

#%% Ped Locations 
import time
simdate = time.strftime("%b %Y")
# lOCATION AND TIME 
latitude = model_inputs.latitude[0]
longitude = model_inputs.longitude[0]
casetime = model_inputs.time[0]
albedogr= model_inputs.ground_albedo[0]
utc_offset=model_inputs.timezone[0]
print 'Location Lat',latitude,'Long',longitude, 'Time ',casetime, 'timezone' ,utc_offset, 'UTC'

# When calculating spatial variation of thermal comfort (i.e. not temporal variation), solar parameters only need to be calculated once
time1 = time.clock()
solarparam = thermalcomfort.calc_solarparam(casetime, latitude, longitude, utc_offset, albedogr, human=True, TC=0)

cases = [myexperiment]
for config in cases:
    #Initializing... 
    pedkeys  = config['pedkeys'] #pedkeys are the coordinates at which Tmrt will be calculated.
    compound = config['model']
    pdTs = config["Tsurf"]
    pdReflect = config["Refl"]
    #The air temperature provided is a bulk value. Repeat the value across all coordinates to create a pdcoord for Tair. 
    pdTa = thermalcomfort.pdcoords_from_pedkeys(pedkeys, np.array([config["Tair"]]*len(pedkeys))) 
    
    #initialize a pdcoord for Tmrt that is filled with zeros
    config['TMRT'] =thermalcomfort.pdcoords_from_pedkeys(pedkeys) 
    for index, row in config['TMRT'].data.iterrows(): #For each pedestrian coordinate...  
        pedkey = (row.x,row.y,row.z) #retrieve the pedestrian's coordinate
        #see code in part 2 of thermalcomfort.py to see step-by-step explanation of the calculation.        
        results = thermalcomfort.all_mrt(pedkey,compound,pdTa,pdReflect,pdTs,solarparam,model_inputs,ped_properties,gridsize=3) #this calculates all steps necessary for MRT calculation.
        row.v = results.TMRT[0]
    #Save results to a csv file    
    config['TMRT'].data.to_csv(config['name']+ '_'+simdate +'_TMRT.csv')
    config['TMRT'].scatter3d()
time2 = time.clock()
print 'TOTAL CALCULATION TIME: ',(time2-time1), 'seconds'

