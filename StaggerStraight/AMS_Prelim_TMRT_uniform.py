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

#%% Ped Locations grid = 2


#%% Initializing MRT calculation - Simluation A1

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

#%% Trying different
[A0,A1,A2]
for config in cases:
    pedkeys  = config['pedkeys']
    compound = config['model']
    pdTa = thermalcomfort.pdcoords_from_pedkeys(pedkeys, np.array([config["Tair"]]*len(pedkeys))) 
    pdTs = config["Tsurf"]
    avgTs = pdTs.data.v.mean()
    pdReflect = config["Refl"]
    avgRs = pdReflect.data.v.mean()
    
    config['TMRT'] =thermalcomfort.pdcoords_from_pedkeys(pedkeys)
    config['TMRT_simpsurf'] =thermalcomfort.pdcoords_from_pedkeys(pedkeys)
    gridsize=1.5
    RH = 50
# For saving simulations
    sim_SVF = thermalcomfort.pdcoords_from_pedkeys(pedkeys) #
    sim_Elong= thermalcomfort.pdcoords_from_pedkeys(pedkeys)
    sim_Eshort= thermalcomfort.pdcoords_from_pedkeys(pedkeys)
    
    for index, row in config['TMRT'].data.iterrows():
        time1 = time.clock()
        pedkey = (row.x,row.y,row.z)
#        results = thermalcomfort.all_mrt(pedkey,compound,pdTa,pdReflect,pdTs,solarparam,model_inputs,ped_constants,gridsize=1.5)
        
        Esky = calc_Esky_emis(np.mean(pdTa.val_at_coord(pedkey).v), RH)
        svf, gvf, intercepts = fourpiradiation(pedkey, compound) #interceptped
        shadowint = check_shadow(pedkey, compound,solarparam.solarvector[0])
   
        SurfTemp, SurfReflect = [call_values(intercepts, surfpdcoord, gridsize) for surfpdcoord in [pdTs, pdReflect]]
        if np.isnan(SurfTemp).any():
            print 'Warning: ' , sum(np.isnan(SurfTemp)), ' intercepts do not have values. Treated as Sky'
            SurfTemp = SurfTemp[~np.isnan(SurfTemp)]
            SurfReflect = SurfReflect[~np.isnan(SurfReflect)]
            svf+=sum(np.isnan(SurfTemp))/Ndir

        SurfAlbedo, SurfEmissivity =  [[x]*len(SurfTemp) for x in [model_inputs.wall_albedo[0], model_inputs.wall_emissivity[0]]] #instead of call values
        Elwall, Eswall = calc_radiation_from_values(SurfTemp, SurfReflect, SurfEmissivity)
        Eground = model_inputs.ground_emissivity[0]*sigma*gvf/2*model_inputs.groundtemp[0]**4
    
        results =  meanradtemp(Esky,Elwall, Eground,Eswall, solarparam, svf,gvf, ped_constants.body_albedo[0], shadow=shadowint)
        time2 = time.clock()
        # AVERAGE REFLECT
       
#        SurfTemp, SurfReflect =  [[x]*len(intercepts) for x in [avgTs,avgRs]]
#        Elwall, Eswall = calc_radiation_from_values(SurfTemp, SurfReflect, SurfEmissivity)
#        TMRT2 =  meanradtemp(Esky,Elwall, Eground,Eswall, solarparam, svf,gvf, ped_constants.body_albedo[0], shadow=shadowint)
#
        sim_SVF.data.v.iloc[index]=  svf
        row.v = results.TMRT[0]
        sim_Elong.data.v.iloc[index]=  results.Elong[0]
        sim_Eshort.data.v.iloc[index]=  results.Eshort[0]
#        config['TMRT_simpsurf'].data.v.iloc[index] = float(TMRT2)

        time2 = time.clock()
        tottime = (time2-time1)/60.0
        print  index, results.TMRT[0], tottime
#        print solarparam
    
    #config['TMRT_simpsurf'].data.to_csv(config['name'] +  '_'+simdate +'_TMRT_simpsurf.csv')
    config['TMRT'].data.to_csv(config['name'] +  '_'+simdate +'_TMRT.csv')
    sim_SVF.data.to_csv(config['name'] +  '_'+simdate +'_SVF.csv')
    sim_Elong.data.to_csv(config['name'] +  '_'+simdate +'_Long.csv')
    sim_Eshort.data.to_csv(config['name'] +  '_'+simdate +'_Long.csv')

    #sim_Elong_simp.data.to_csv(config['name'] +  '_'+simdate +'_Long.csv')

