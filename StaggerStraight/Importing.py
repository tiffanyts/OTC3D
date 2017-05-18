# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:46:47 2017
Importing the correct files
@author: Not Tiffany
"""


import os
import datetime
import numpy as np
import scipy.io
import pandas as pd

#import ExtraFunctions 
#import thermalcomfort 
#import pyliburo

#%%
simdate = 'Mar17'

current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
folderpath = os.path.join(parent_path, "StaggerStraight","Outputs_"+simdate)
#%% SET Input

ped_properties = pd.DataFrame({
    'height':[1.5],
    'skin_wetness':[0.088],
    'mass': [70], #height
    'height':[1.7], #meters
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
A1 = {"name":"Alb0.3_WWR0.4_SHGC0.2_AR1_Lp0.25","canyon":32,"cube":32,"Lp":0.25,"AR":1., "hour" :13,"albedo":0.3,"gridsize":0.125}
A2 = {"name":"Alb0.3_WWR0.4_SHGC0.2_AR1_Lp0.44","canyon":16,"cube":32,"Lp":0.44,"AR":2., "hour" : 13,"albedo":0.3,"gridsize":0.25}

B0 = {"name":"Alb0.1_WWR0.4_SHGC0.2_AR1_Lp0.0625","canyon":96,"cube":32,"Lp":0.0625,"AR":0.33, "hour" : 13, "albedo":0.1,"gridsize":0.125}
B1 = {"name":"Alb0.1_WWR0.4_SHGC0.2_AR1_Lp0.25","canyon":32,"cube":32,"Lp":0.25,"AR":1., "hour" :13,"albedo":0.1,"gridsize":0.125}
B2 = {"name":"Alb0.1_WWR0.4_SHGC0.2_AR1_Lp0.44","canyon":16,"cube":32,"Lp":0.44,"AR":2., "hour" : 13,"albedo":0.1,"gridsize":0.25}
#%%

cases = [A0,A1,A2,B0,B1,B2]#]#
for config in cases:
    config["model"] = ExtraFunctions.makemodelmatrix((5,3),config['canyon']*config["gridsize"],config['cube']*config['gridsize'],config['cube']*config['gridsize'])["model"]
    config["square"] = ExtraFunctions.make_sq_center(pyliburo.py3dmodel.calculate.get_centre_bbox(config["model"]),(config["canyon"]*config['gridsize']+config["cube"]*config['gridsize']-1)/2)
    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square'])
    config["lil_square"] = ExtraFunctions.make_sq_center(pyliburo.py3dmodel.calculate.get_centre_bbox(config["model"]),(config["cube"]*config['gridsize'])/2,)
    #config['pedkeys']  = np.loadtxt('pedkeys\\'+config['name']+'_pedkeys_HD')
    #config['pedkeys'] = zip(np.arange( a[0],c[0],0.5),np.arange( a[1],c[1],0.5),np.array([1.5]*len(np.arange( a[0],c[0],0.5))),np.arange( a[1],c[1],0.5))
        
    config['pedkeys'] = np.array([(x,y,1.5) for x in np.linspace( a[0],c[0],25) for y in np.linspace( a[1],c[1],25)])

    #config['pedkeys'] = np.array([ [x,y,z] for [x,y,z] in config['pedkeys'] if (((x >= a[0]) & (x <= c[0])) & ((y >= a[1]) &(y <= c[1]) ))  ])
    l,m,n,o= pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['lil_square'])
    config['pedkeys'] = np.array([ [x,y,z] for [x,y,z] in config['pedkeys'] if not (((x > l[0]-config['gridsize']) & (x < n[0]+config['gridsize'])) & ((y > l[1]-config['gridsize']) &(y < n[1]+config['gridsize']) ))  ])
#for config, parallel in zip(cases, [A0,A1,A2]):
#    config['pedkeys'] = parallel['pedkeys']
          
#%% RAYMAN OBSTACLES
#ped_min = (21.25, 38.0,1.5)
#ped_max = (25.0, 42.25,1.5)
#vertices = [[(point.X()-ped_max[0],point.Y()-ped_max[1],point.Z()) for point in pyliburo.py3dmodel.fetch.points_frm_solid(solid)] for solid in pyliburo.py3dmodel.fetch.solids_frm_compsolid(A0["model"])]       
#for line in vertices: 
#    print  '\ng',
#    for (X,Y,Z) in line[1::2]: 
#        print X,Y,Z,
#    for (X,Y,Z) in line[::2]: 
#        print X,Y,Z,

#for config in [A1]:
#    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square'])
#    x = np.arange(a[0],c[0]+0.25,0.25)
#    y = np.arange(a[1],c[1]+0.25,0.25)
#    pedlocations = [(x[i],y[j],1.5) for i in range(len(x)) for j in range(len(y)) if pyliburo.py3dmodel.calculate.point_in_solid(config['model'],(x[i],y[j],1.5)) is False]
#    config['pedkeys'] = np.array(pedlocations)
#    np.savetxt(config['name']+'_pedkeys_HD',config['pedkeys'])    

#for config in [S0,S1,S2]:
#    config["model"] = makemodelstagger((6,3),config['canyon'],config['cube'],config['cube'])["model"]
#    midpoint = ((config['canyon']+config['cube'])*1.5+config['cube']/2,(config['canyon']+config['cube'])*2+config['cube']/2,1.5)
#    
#    config["square"] = make_sq_center(midpoint,(config["canyon"]+config["cube"])/2)

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
A1['wind_input']  = testx['s'][0][0][1]
A2['wind_input']  = testx['s'][0][0][2]
B0['wind_input'] = testx['s'][0][0][0]
B1['wind_input']  = testx['s'][0][0][1]
B2['wind_input']  = testx['s'][0][0][2]

#S0['wind_input']  = testx['s'][0][0][3]
#S1['wind_input'] = np.fliplr(testx['s'][0][0][6])
#S2['wind_input']  = testx['s'][0][0][8]
#%% Processing wind - A0

for config in cases:
    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square'])
    dim = int(config['wind_input'].shape[0])
    X,Y = np.meshgrid(np.linspace(a[0],c[0],dim),np.linspace(a[1],c[1],dim))
    Z = np.array([1.5]*len(X.flatten()))
    windkeys = np.vstack([X.flatten(),Y.flatten(),Z])
    config['wind'] = thermalcomfort.pdcoords_from_pedkeys(windkeys.T,abs(config['wind_input'].flatten()))
    
##%% Stagger
#for config in []: #S0,S1,S2
#    Tmrt = config['TMRT']
#    a,b,c,d = pyliburo.py3dmodel.fetch.pyptlist_frm_occface(config['square'])
#    
#    BL = (a[0]-config['canyon']/2-config['cube'], a[1]+config['cube']/2, 1.5)
#    
#    Z = np.ma.array(config['wind_input'])
#    
#    Zt = np.tile(Z, 2)     
#    Zt = np.tile(Zt, (2,1))
#    
#    
#    Zt = Zt.transpose()
#    Ydim,Xdim = Zt.shape
#    xt = np.arange(BL[0],BL[0]+Xdim); yt = np.arange(BL[1]-Ydim/2,BL[1]+Ydim/2)
#    Xt,Yt = np.meshgrid(xt, yt)
#    
#    plt.figure()    
#    plt.contourf(Xt,Yt,Zt)
##    vertices = [(a[0]+config['canyon']/2, a[1]+config['canyon']/2),
##                (b[0]+config['canyon']/2, b[1]-config['canyon']/2),
##                (c[0]-config['canyon']/2, c[1]-config['canyon']/2),
##                (d[0]-config['canyon']/2, d[1]+config['canyon']/2)]
#                
#    vertices = [(Tmrt.data.x.min()+config['canyon']/2, Tmrt.data.y.min()+config['canyon']/2),
#                (Tmrt.data.x.max()-config['canyon']/2, Tmrt.data.y.min()+config['canyon']/2),
#                (Tmrt.data.x.max()-config['canyon']/2, Tmrt.data.y.max()-config['canyon']/2),
#                (Tmrt.data.x.min()+config['canyon']/2, Tmrt.data.y.max()-config['canyon']/2)]
#    shape = patches.PathPatch(Path(vertices), facecolor='white', lw=0)
#    plt.gca().add_patch(shape) 
#    
#    
#    windkeys= np.array(zip(Xt.flatten(),Yt.flatten(),[1.5]*len(Xt.flatten())))
#    wind = Zt.flatten()
#
#    config['wind']  = pdcoords_from_pedkeys(windkeys,wind)
##
#    print config['name']
#%% Importing TMRT (IF ALREADY CALCULATED)
#simdate = 'Mar4_HD'
#
#for config in cases:
#    config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +'_'+  simdate+'_TMRT.csv'))
#    config['TMRT'].data.v = config['TMRT'].data.v-273.15
#    config['SVF'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_SVF.csv'))

#%%
simdate = 'Mar_11'
##
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
folderpath = os.path.join(parent_path, "StaggerStraight","Outputs_"+simdate)
for config in [A0,A1,A2]:
     config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,'TMRTSET',config['name'] +  '_'+simdate+'_TMRT.csv'))
     config['TMRT_Ts'] = thermalcomfort.pdcoord(os.path.join(folderpath,'TMRTSET',config['name'] +  '_'+simdate+'_TMRT_simpsurf.csv'))
     config['SET'] = thermalcomfort.pdcoord(os.path.join(folderpath,'Auciliem',config['name'] +  '_'+simdate+'_SET_Auciliem.csv'))
#
#simdate = 'Mar4_HD'
#current_path = os.path.dirname("__file__")
#parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
#folderpath = os.path.join(parent_path, "StaggerStraight","Outputs_"+simdate)
#
#for config in [A0,A1,A2]:
#    config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_TMRT.csv'))
#    config['TMRT'].data.v = config['TMRT'].data.v-273.15
#    #config['TMRT_Ts'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_TMRT_simpsurf.csv'))
#%%
simdate = 'Mar4_HD'
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
folderpath = os.path.join(parent_path, "StaggerStraight","Outputs_"+simdate)

for config in [A0,A1,A2]:
    config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_TMRT.csv'))
    
simdate = 'Mar13_alb'
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
folderpath = os.path.join(parent_path, "StaggerStraight","Outputs_"+simdate)

for config in [B0,B1,B2]:
    config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_TMRT.csv'))
    config['TMRT_Ts'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_TMRT_simpsurf.csv'))
    config['SET'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_SET.csv'))   