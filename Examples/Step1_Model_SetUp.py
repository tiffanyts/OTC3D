# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:46:47 2017
Importing the correct files
@author: Tiffany Sin 2017 
"""
import os
import datetime
import numpy as np
import scipy.io
import pandas as pd
import ExtraFunctions 
import thermalcomfort 
import pyliburo
import pandas as pd

#%%
simdate = 'Today'
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
folderpath = os.path.join(parent_path,"Outputs_"+simdate) #location to save results
#%% SET Input

ped_properties = pd.DataFrame({
    'height':[1.5],
   'skin_wetness':[0.088],
    'mass': [70], #height
    'height':[1.5], #meters
    'eff_radiation_SA_ratio':[0.73], #AR - ratio of effective radiation area of body and body surf. area(Fanger 1967)
    'body_emis':[0.95], #average emissivity of clothing/body surface, (ASHRAE 1997 8.7)
    'body_albedo':[0.3],
    'met':[1.2], #metabolic rate
    'work':[0], #[W]
    'iclo':[0.34], # icl, clothing vapor permeation efficiency (ASHRAE 8.8)
    'icl':[0.36],  #insulation of air layer. 0.36*k for shorts and t-shirt (ASHRAE 8.8) 
    'fcl':[1.1],    #clothing area factor. 1.1 for working shorts, short-sleeved shirt,
   })


#pedproperties_pathfile = raw_input('Enter the full path of the pedestrain properties input file ')
#ped_properties=pd.read_csv(pedproperties_pathfile)

#modelinputs_pathfile = raw_input('Enter the full path of the model input file ')
#model_inputs=pd.read_csv(modelinputs_pathfile)  
 
model_inputs = pd.DataFrame({
    'latitude':[1.383419],
    'longitude':[103.902707],
    'time':[(2016,7,6,12,00)],
    'wall_albedo':[0.3],
    'wall_emissivity': [0.90],
    'ground_emissivity':[0.95],
    'ground_albedo':[0.30],
    'groundtemp': [302],
   # 'surftemp':[301.2], #approx from jimeno's calulation

    })


myexperiment = {"name":"Example","canyon":96,"cube":32,"AR":0.33, "albedo":0.3, "gridsize":0.125}
#%% Building the 3D model ('model') and calculating coordinates along the pedestrian grid ('pedkeys'). 

cases = [myexperiment]
for config in cases:
    #1 build the 3D model - see other example for different methods. This experiment is for a matrix of cubic building.
    config["model"] = ExtraFunctions.makemodelmatrix((6,4),config['canyon']*config["gridsize"],config['cube']*config['gridsize'],config['cube']*config['gridsize'])["model"]

    #2 define the area of study. In this case, the pedestrian grid is within a square around a central building. 
    config["square"] = ExtraFunctions.make_sq_center(pyliburo.py3dmodel.calculate.get_centre_bbox(config["model"]),(config["canyon"]*config['gridsize']+config["cube"]*config['gridsize']-1)/2)
    
    #3 calculate the coordinates of the pedestrian keys.
    # 3a) coordinates within your area of study. 'square' is an outer boundary
    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square']) 
    config['pedkeys'] = np.array([(x,y,ped_properties.height[0]) for x in np.linspace( a[0],c[0],25) for y in np.linspace( a[1],c[1],25)])
    config['pedkeys'] = np.array([ [x,y,z] for [x,y,z] in config['pedkeys'] if (((x >= a[0]) & (x <= c[0])) & ((y >= a[1]) &(y <= c[1]) ))  ])
    
    # 3b) remove coordinates inside of the central building. 'lil_square' is an inner boundary.
    config["lil_square"] = ExtraFunctions.make_sq_center(pyliburo.py3dmodel.calculate.get_centre_bbox(config["model"]),(config["cube"]*config['gridsize'])/2,) #area of building 
    l,m,n,o= pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['lil_square'])
    config['pedkeys'] = np.array([ [x,y,z] for [x,y,z] in config['pedkeys'] if not (((x > l[0]-config['gridsize']) & (x < n[0]+config['gridsize'])) & ((y > l[1]-config['gridsize']) &(y < n[1]+config['gridsize']) ))  ])

#%% Importing Thermal data. The sample data here was retrived from the results of a TUFIOBES model of the same building geometry. 

#run Importing_TUFIOBES first
for config in cases:
    therm_input = pd.read_csv(os.path.join(parent_path,'Examples','Input_Data',config["name"]+'_surface_data.csv'),delimiter=",",usecols=(2,3,4,6,7,8))
#    
    config['Tsurf_ground'] = thermalcomfort.pdcoord(therm_input[therm_input['0']=='ground'][['x','y','z','temp']])
    config['Tsurf'] = thermalcomfort.pdcoord(therm_input[['x','y','z','temp']])
    config['Refl_ground'] = thermalcomfort.pdcoord(therm_input[therm_input['0']=='ground'][['x','y','z','refl']])
    config['Refl'] = thermalcomfort.pdcoord(therm_input[['x','y','z','refl']]) 
    # This imports the data into pdcoord form. Look at it in 3D: 
    config['Tsurf'].scatter3d()
    
#However, the data provided is only for one building. In our experiment, these buildings are repeated in a matrix. Also, it needs to be aligned to the 3D model, which is centered at the origin. 
for config in cases:
    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square']) #retrieve the coordinates of the 3D model.
    config['Tsurf'] = config['Tsurf'].recenter(origin=(a[0],a[1])) 
    config['Refl'] = config['Refl'].recenter(origin=(a[0],a[1]))
    config['Tsurf'] = config['Tsurf'].repeat_outset() #repeat the matrix in all 8 directions. 
    config['Refl'] = config['Refl'].repeat_outset()
    config['Tsurf'] = config['Tsurf'].repeat_outset()
    config['Refl'] = config['Refl'].repeat_outset() 
    #config['Tsurf'].scatter3d() to look at it again - warning: rendering may be slow!
##    
for config in cases:
    #Detailed spatial air temperature is unavailable for this example, so a bulk temperature is taken from an energy balance model (TUFIOBES) for the time of day of this experiment. 
   # hour = pd.DatetimeIndex([pd.to_datetime(model_inputs.time)]).hour[0]
   #otherthermal = pd.read_csv(casefolder+'\\Tsfc_Facets.out',usecols=[5,6,15],header=None,sep='\s+',names=['day','hour','temperature'])
   #otherthermal['hour']= otherthermal['hour'].apply(np.round)
   #config['Tair'] = np.mean(otherthermal[otherthermal['hour']==model_inputs['time'][0][3]]['temperature'])    
    otherthermal = pd.read_csv(os.path.join(parent_path,'Examples','Input_Data','Example_thermal_data.out'),usecols=[5,6,15],header=None,sep='\s+',names=['day','hour','temperature'])
    otherthermal['hour']= otherthermal['hour'].apply(np.round)
    config['Tair'] = np.mean(otherthermal[otherthermal['hour']==model_inputs['time'][0][3]]['temperature'])

#%% Importing Wind. The sample data here was taken from a CFD model of the same geometry. The data was provided as a matrix, as opposed to coordinates, according to the mesh grid.  

myexperiment['wind_input'] = np.loadtxt(os.path.join(parent_path,'Examples','Input_Data','Example_wind_input_matrix'))
for config in cases: 
    # Since the CFD output is provided as a matrix, and not with coordinates, it needs to be 'fitted' to the area of study and converted into pdcoord.   
    dim = int(config['wind_input'].shape[0]) 

    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square']) #determining the coordinates of the area of study
    X,Y = np.meshgrid(np.linspace(a[0],c[0],dim),np.linspace(a[1],c[1],dim)) #calculating coordinates based on the resolution of the grid data. 
     
    Z = np.array([1.5]*len(X.flatten()))
    windkeys = np.vstack([X.flatten(),Y.flatten(),Z]) #listing out these coordiantes 
    config['wind'] = thermalcomfort.pdcoords_from_pedkeys(windkeys.T,abs(config['wind_input'].flatten())) #mapping each data point on the grid to spatial coordinates. 

#%% If you've already calculated mean radiant temperature (Tmrt), import it and skip straight to SET calculations. 
#for config in cases:
#    config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_TMRT.csv'))
    
