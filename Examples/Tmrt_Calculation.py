# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:23:35 2017
TMRT for Staggered and Aligned Matrices
@author: SHARED1-Tiffany

Aligned- A0,A1,A2  - 0.0625		0.25		0.444 
Staggered- S0,S1,S2- 0.0625		0.25		0.444

"""
#import pyliburo
#import ExtraFunctions
import numpy
import datetime
import time
#%%
#import Importing

#%% Ped Locations 
simdate = 'Today'

latitude = model_inputs.latitude[0]
longitude = model_inputs.longitude[0]
casetime = model_inputs.time[0]
timezone = -8
RH = model_inputs.RH[0]

#calculate once

time1 = time.clock()
solarparam = thermalcomfort.solar_param(casetime,latitude,longitude,UTC_diff=timezone,groundalbedo=model_inputs.ground_albedo[0])
time2 = time.clock()
print 'solar time ',(time2-time1)/60.0

cases = [A0]
for config in cases:
    pedkeys  = config['pedkeys']
    compound = config['model']
    pdTa = thermalcomfort.pdcoords_from_pedkeys(pedkeys, np.array([config["Tair"]]*len(pedkeys))) #Tair is constant
    pdTs = config["Tsurf"]
    avgTs = pdTs.data.v.mean()
    pdReflect = config["Refl"]
    avgRs = pdReflect.data.v.mean()
    
    config['TMRT'] =thermalcomfort.pdcoords_from_pedkeys(pedkeys)

    for index, row in config['TMRT'].data.iterrows():
        time1 = time.clock()
        pedkey = (row.x,row.y,row.z)
        results = thermalcomfort.all_mrt(pedkey,compound,pdTa,pdReflect,pdTs,solarparam,model_inputs,ped_constants,gridsize=3)
        row.v = results.TMRT[0]
        time2 = time.clock()
        tottime = (time2-time1)/60.0
        print  index, results.TMRT[0], tottime
    
    config['TMRT'].data.to_csv(config['name'] +  '_'+simdate +'_TMRT.csv')

