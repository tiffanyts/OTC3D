# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 22:37:28 2017
Comparison of shadows with density
@author: Tiffany
"""

#run importing

import numpy
import datetime
import time
import matplotlib.pyplot as plt


latitude = model_inputs.latitude[0]
longitude = model_inputs.longitude[0]
casetime = model_inputs.time[0]
timezone = -8

#calculate once for area
#[sunpx, sunpy,sunpz, solarvf, E_sol, Sky_Diffuse, Ground_Diffuse]
time1 = time.clock()
solarparam = thermalcomfort.solar_param(casetime,latitude,longitude,UTC_diff=timezone,groundalbedo=model_inputs.ground_albedo[0])
time2 = time.clock()
print 'solar time ',(time2-time1)/60.0
#shadow = get_shadow(pedkeys, compound,(sunpx, sunpy, sunpz))
#%%
cases = [A0,A1,A2]

for config in cases:
#    compound = config['model']
#    total = len(config['Tsurf'].data)
#    config['sunct'] = 0.
#    config['sunlit'] = thermalcomfort.pdcoord(config['Tsurf'].data)
#
#    for index, row in config['sunlit'].data.iterrows():
#        surfkey = (row.x,row.y,row.z)
#        shadow = float(thermalcomfort.check_shadow(surfkey, compound,solarparam.solarvector[0]))
#        config['sunct'] += shadow/total
#        row.v = shadow
    config['groundsunct'] = sum(config['sunlit'].data[config['sunlit'].data.z==0].v)/len(config['sunlit'].data[config['sunlit'].data.z==0])
    config['wallsunct'] = sum(config['sunlit'].data[config['sunlit'].data.z!=0].v)/len(config['sunlit'].data[config['sunlit'].data.z!=0])
#%%
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height-7,
                '%.1f' % height,
                ha='center', va='bottom',color='black',fontsize = 13)
        
#%%
fig, ax = plt.subplots()
width = 0.4
wallbar = ax.bar(range(3), [100*x for x in [A0['wallsunct'],A1['wallsunct'],A2['wallsunct']]], width,label = 'Walls', color='navajowhite')
grdbar = ax.bar(np.arange(width,3.2,1), [100*x for x in [A0['groundsunct'],A1['groundsunct'],A2['groundsunct']]], width,label='Ground',color='orange')
ax.set_xticklabels(map(str, [A0['Lp'],A1['Lp'],A2['Lp']]),fontsize = 12)
ax.set_xticks(np.arange(width/2,3.2,1))
#ax.set_yticklabels(fontsize = 12)
ax.legend(fontsize = 15)
ax.set_title('Proportion of Surface in Sunlight',  fontsize = 20)
ax.set_xlabel(r'Urban Density ($\lambda_{p}$)',  fontsize = 15)
ax.set_ylabel(r'Sunlit Area / Total Area (%)', fontsize = 15)

autolabel(wallbar)
autolabel(grdbar)