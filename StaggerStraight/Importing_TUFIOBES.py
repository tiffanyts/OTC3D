# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:10:00 2017
Importing TUFIOBES specifically
@author: SHARED1-Tiffany

Combine all 10 surfaces
Create a pandas of hour, coord, Ts, reflect rad.
Create a pandas of hour, Ta. 

"""
import os 
import numpy as np
import pandas as pd 
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

A0 = {"name":"Alb0.3_WWR0.4_SHGC0.2_AR1_Lp0.0625","canyon":96,"cube":32,"Lp":0.0625,"AR":0.33, "hour" : 13, "albedo":0.3}
A1 = {"name":"Alb0.3_WWR0.4_SHGC0.2_AR1_Lp0.25","canyon":32,"cube":32,"Lp":0.25,"AR":1., "hour" :13,"albedo":0.3}
A2 = {"name":"Alb0.3_WWR0.4_SHGC0.2_AR1_Lp0.44","canyon":16,"cube":32,"Lp":0.44,"AR":2., "hour" : 13,"albedo":0.3}
B0 = {"name":"Alb0.1_WWR0.4_SHGC0.2_AR1_Lp0.0625","canyon":96,"cube":32,"Lp":0.0625,"AR":0.33, "hour" : 13, "albedo":0.1}
B1 = {"name":"Alb0.1_WWR0.4_SHGC0.2_AR1_Lp0.25","canyon":32,"cube":32,"Lp":0.25,"AR":1., "hour" :13,"albedo":0.1}
B2 = {"name":"Alb0.1_WWR0.4_SHGC0.2_AR1_Lp0.44","canyon":16,"cube":32,"Lp":0.44,"AR":2., "hour" : 13,"albedo":0.1}

#%% Reading one file 
newsurf= []
for config in [A0,A1,A2,B0,B1,B2]:
    surfs= [] ;
    casefolder = os.path.join(parent_path, "TUF-IOBES Results for OTC3D",config["name"])
    for i in range(1,10):
        surfflux = pd.read_csv(casefolder+'\\fort.'+str(100+i),header=None,sep='\s+', index_col = 0, names=['hour','x','y','z','flux'])
        surftemp = pd.read_csv(casefolder+'\\fort.'+str(200+i),header=None,sep='\s+', index_col = 0, usecols=[0,5], names=['index','temp'])
        surfrefl = pd.read_csv(casefolder+'\\fort.'+str(300+i),header=None,sep='\s+', index_col = 0, usecols=[0,5], names=['index','refl'])

        if i == 1:
            newsurf = pd.concat([surfflux, surftemp, surfrefl,pd.Series('ground', index=surftemp.index.values)], axis=1,join_axes=[surfflux.index])
        elif i in [2,3,4,5]:
            newsurf = pd.concat([surfflux, surftemp, surfrefl,pd.Series('wall', index=surftemp.index.values)], axis=1,join_axes=[surfflux.index])
        elif i in [6,7,8,9]:
            newsurf = pd.concat([surfflux, surftemp, surfrefl,pd.Series('window', index=surftemp.index.values)], axis=1,join_axes=[surfflux.index])

        newsurf['hour']= newsurf['hour'].apply(np.round)
        if not np.all(newsurf['hour'].unique()==np.arange(1.,24.)): print 'NOT RIGHT'
        surfs.append(newsurf); 

    allsurf = pd.concat(surfs)
    for hour in np.arange(1.,24.):
        allsurf[allsurf['hour']==hour].to_csv(os.path.join(casefolder,'Sorted',config["name"]+'SurfaceProperties_'+str(int(hour))))
