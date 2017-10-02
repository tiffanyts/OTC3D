# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:43:35 2017
@author: Tiffany Sin, Negin Nazarian 2017

After calculating TMRT and importing initial information, SET can be calculated at every pedestrian location. This example is dependent on previous steps. 
"""
import time
import numpy as np
import thermalcomfort

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
#%% 
#import Importing
simdate = time.strftime("%b %Y")

#%% Microclimate
cases = [myexperiment]
time1 = time.clock()
for config in cases:
    Tmrt = config['TMRT']
    pedkeys  = config['pedkeys'] #imported previously. 
    Ta = config['Tair'] #constant
    V = config['wind'] 
    config['SET']  = thermalcomfort.pdcoords_from_pedkeys(pedkeys) #initialize a pdcoord for SET that is filled with zeros

    for index,row in config['SET'].data.iterrows(): #calculate SET line by line along the pdcoord (i.e. for each coordinate on the grid)
        time1 = time.clock()
        pedkey = (row.x,row.y,row.z)
        microclimate = pd.DataFrame({
        'T_air':[Ta],
        'wind_speed':[np.mean(abs(V.val_at_coord(pedkey, radius = 0.2).v))],
        'mean_radiant_temperature': [np.mean(Tmrt.val_at_coord(pedkey, radius = 1).v)],
        'RH':[model_inputs.RH[0]],  
        })
    
        row.v = thermalcomfort.calc_SET(microclimate,ped_properties)    
        time2 = time.clock()
        tottime = (time2-time1)/60.0
       # print  ' row ' , index,  ' | SET is ', row.v ,'. TIME TAKEN', tottime

    config['SET'].data.to_csv(config['name'] + '_'+simdate +'_SET.csv')
    config['SET'].scatter3d()

time2 = time.clock()
print 'TOTAL CALCULATION TIME: ',(time2-time1), 'minutes'
