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
    
model_inputs = pd.DataFrame({
    'latitude':[1.383419],
    'longitude':[103.902707],
    'time':[(2016,7,6,12,00)],
    'wall_albedo':[0.3],
    'wall_emissivity': [0.90],
    'ground_emissivity':[0.95],
    'ground_albedo':[0.30],
    'groundtemp': [302],
    #'surftemp':[301.2], #approx from jimeno's calulation

    })

A0 = {"name":"Alb0.3_WWR0.4_SHGC0.2_AR1_Lp0.0625","canyon":96,"cube":32,"Lp":0.0625,"AR":0.33, "hour" : 13, "albedo":0.3, "gridsize":0.125}

cases = [A0]
for config in cases:
    config["model"] = ExtraFunctions.makemodelmatrix((5,3),config['canyon']*config["gridsize"],config['cube']*config['gridsize'],config['cube']*config['gridsize'])["model"]
    config["square"] = ExtraFunctions.make_sq_center(pyliburo.py3dmodel.calculate.get_centre_bbox(config["model"]),(config["canyon"]*config['gridsize']+config["cube"]*config['gridsize']-1)/2)
    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square'])
    config["lil_square"] = ExtraFunctions.make_sq_center(pyliburo.py3dmodel.calculate.get_centre_bbox(config["model"]),(config["cube"]*config['gridsize'])/2,) #area of building 
    config['pedkeys'] = np.array([(x,y,ped_properties.height[0]) for x in np.linspace( a[0],c[0],25) for y in np.linspace( a[1],c[1],25)])
    config['pedkeys'] = np.array([ [x,y,z] for [x,y,z] in config['pedkeys'] if (((x >= a[0]) & (x <= c[0])) & ((y >= a[1]) &(y <= c[1]) ))  ]) #remove area of building
    l,m,n,o= pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['lil_square'])
    config['pedkeys'] = np.array([ [x,y,z] for [x,y,z] in config['pedkeys'] if not (((x > l[0]-config['gridsize']) & (x < n[0]+config['gridsize'])) & ((y > l[1]-config['gridsize']) &(y < n[1]+config['gridsize']) ))  ])

#%% Importing Thermal data (TUFIOBES)
#run Importing_TUFIOBES first
for config in cases:
    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square'])
    
    casefolder = os.path.join(parent_path, "TUF-IOBES Results for OTC3D",config["name"])
    therm_input = pd.read_csv(os.path.join(casefolder,'Sorted',config["name"]+'SurfaceProperties_'+str(int(config['hour']))),delimiter=",",usecols=(2,3,4,6,7,8))
#    
    config['Tsurf_ground'] = thermalcomfort.pdcoord(therm_input[therm_input['0']=='ground'][['x','y','z','temp']])
    config['Tsurf'] = thermalcomfort.pdcoord(therm_input[['x','y','z','temp']])
    config['Refl_ground'] = thermalcomfort.pdcoord(therm_input[therm_input['0']=='ground'][['x','y','z','refl']])
    config['Refl'] = thermalcomfort.pdcoord(therm_input[['x','y','z','refl']]) 
    
    config['Tsurf'] = config['Tsurf'].recenter(origin=(a[0],a[1]))
    config['Refl'] = config['Refl'].recenter(origin=(a[0],a[1]))
    config['Tsurf'] = config['Tsurf'].repeat_outset()
    config['Refl'] = config['Refl'].repeat_outset()
    config['Tsurf'] = config['Tsurf'].repeat_outset()
    config['Refl'] = config['Refl'].repeat_outset() 
##    
    otherthermal = pd.read_csv(casefolder+'\\Tsfc_Facets.out',usecols=[5,6,15],header=None,sep='\s+',names=['day','hour','temperature'])
    otherthermal['hour']= otherthermal['hour'].apply(np.round)
    config['Tair'] = np.mean(otherthermal[otherthermal['hour']==model_inputs['time'][0][3]]['temperature'])
#%% Importing Wind
testx = scipy.io.loadmat(os.path.join(parent_path,"StaggerStraight",'VelocityAtPedLevel_LES_LpStaAlig.mat'))

A0['wind_input'] = testx['s'][0][0][0] 
for config in cases: #process CFD output
    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square'])
    dim = int(config['wind_input'].shape[0])
    X,Y = np.meshgrid(np.linspace(a[0],c[0],dim),np.linspace(a[1],c[1],dim))
    Z = np.array([1.5]*len(X.flatten()))
    windkeys = np.vstack([X.flatten(),Y.flatten(),Z])
    config['wind'] = thermalcomfort.pdcoords_from_pedkeys(windkeys.T,abs(config['wind_input'].flatten())) 
    

for config in [A0,A1,A2]:
    config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_TMRT.csv')) #read CSV of mean radiant temperature if already calculated
    
