# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:43:35 2017
AMS_Prelim_Output
@author: SHARED1-Tiffany
"""
import time
import numpy

import matplotlib.pyplot as plt

from matplotlib.path import Path
import matplotlib.patches as patches
#%% 
#import Importing
simdate = 'Today'


#%% Microclimate

for config in cases:
    Tmrt = config['TMRT']
    pedkeys  = config['pedkeys']
    Ta = config['Tair'] #constant

    V = config['wind'] 
    RH = 50
    config['SET']  = thermalcomfort.pdcoords_from_pedkeys(pedkeys)

    for index,row in config['SET'].data.iterrows():
        time1 = time.clock()
        pedkey = (row.x,row.y,row.z)
        microclimate = pd.DataFrame({
        'T_air':[Ta],
        'wind_speed':[np.mean(abs(V.val_at_coord(pedkey, radius = 0.2).v))],
        'mean_radiant_temperature': [np.mean(Tmrt.val_at_coord(pedkey, radius = 1).v)-273.15],
        'RH':[50],  
        })
    
        row.v = thermalcomfort.calc_SET(microclimate,ped_properties)    
        time2 = time.clock()
        tottime = (time2-time1)/60.0
        print 'Lp :  ', config['Lp'], '|  row ' , index,  ' | SET is ', row.v ,'. TIME TAKEN', tottime

    config['SET'].data.to_csv(config['name'] + '_'+simdate +'_SET.csv')



