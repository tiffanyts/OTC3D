# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:28:56 2016

@author: SHARED1-Tiffany
Functions for building simple geometry model
"""
#
import os
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd
import envuo
import pvlib
from pvlib import solarposition as sp
from pvlib import irradiance as ir
from pvlib import clearsky as csky
from OCC.Display import OCCViewer
#from ExtraFunctions import *

def solar_param(time_str,latitude,longitude):
    time = pd.Timestamp(time_str[0], tz=time_str[1])  
    zenith = pvlib.solarposition.get_solarposition(time, latitude,longitude).zenith[0]
    altitude = pvlib.solarposition.get_solarposition(time, latitude,longitude).elevation[0]
    azimuth = pvlib.solarposition.get_solarposition(time, latitude,longitude).azimuth[0]
    sunpz = np.sin(np.radians(altitude)); hyp = np.cos(np.radians(altitude))
    sunpy = hyp*np.cos(np.radians(azimuth))
    sunpx = hyp*np.sin(np.radians(azimuth))
    solar_pmt =  pvlib.clearsky.ineichen(time, latitude, longitude)
    E_sol= solar_pmt.dni[0] #direct normal solar irradiation  [W/m^2]
    GroundReflect_V = pvlib.irradiance.grounddiffuse(90,solar_pmt.ghi,albedo =0.18)[0]  #Ground Reflected Solar Irradiation - vertical asphalt surface [W/m^2]
    Diffuse_V =  pvlib.irradiance.isotropic(90, solar_pmt.dhi)[0] #Diffuse Solar Irradiation - vertical surface[W/m^2]. isotropic not v accurate
    E_di = GroundReflect_V + Diffuse_V;     
    sin_bita=(sunpz/((sunpx)**2+(sunpy)**2+(sunpz)**2)**(0.5));
    cos_bita=(sunpz/((sunpx)**2+(sunpy)**2+(sunpz)**2)**(0.5));
    sin_alpha=abs(sunpx)/((sunpx)**2+(sunpy)**2)**(0.5);
    cos_alpha=abs(sunpy)/((sunpx)**2+(sunpy)**2)**(0.5);
    solarvf=0.0355*sin_bita+2.33*cos_bita*(0.0213*cos_alpha**2+0.00919*sin_alpha**2)**(0.5);      
    return zenith,altitude,azimuth,  sunpx, sunpy,sunpz,E_sol, E_di, solarvf
    
def get_shadow(pedestrian_keys, model,solar_vector):
    """ Returns a dictionary of shadowed (0) and sunlit (1) locations. Ignores points that are on the wall (treats them as not shadowed)  """
    shadow = {}
    for key in pedestrian_keys:
        occ_interpt, occ_interface = envuo.py3dmodel.calculate.intersect_shape_with_ptdir(model,key,solar_vector)
        if occ_interpt != None: shadow.update({key: 0}) 
        else: shadow.update({key:1})
    return shadow

#def calc_wvf(pedestrian, key, face, mesh_area): 
#    """ Calculates WVF for a key that is on a face """ 
#    normal = envuo.py3dmodel.construct.make_vector((0,0,0),envuo.py3dmodel.calculate.face_normal(face))
#    ped2surf = envuo.py3dmodel.construct.make_vector(pedestrian, key)
#    return mesh_area*abs(normal.Dot(ped2surf.Normalized()))/ (4*np.pi*ped2surf.Magnitude()**2)

def calc_wvf(pedestrian, key, face, mesh_area,radius): 
    """ Calculates WVF for a key that is on a face """ 
    normal = envuo.py3dmodel.construct.make_vector((0,0,0),envuo.py3dmodel.calculate.face_normal(face))
    surf2ped = envuo.py3dmodel.construct.make_vector(key,pedestrian)
    sa_ped = 4.0*np.pi*radius**2
    theta = normal.Angle(surf2ped)
    h = surf2ped.Magnitude()/radius 
    phi = np.arctan(1/h)
    threshold = np.pi/2.0 - phi

    if abs(h*np.cos(theta)) > 1:
        F = abs(np.cos(theta))/h**2; 
    else:
        x = np.sqrt(h**2-1)/np.tan(theta) #np.sqrt(h**2-1)
        y = np.sqrt(1-x**2) #-x/np.tan(theta) #
        F = (np.pi - abs(np.cos(x)) - x*y*np.tan(theta)**2)*abs(np.cos(theta))/(np.pi*h**2) + np.arctan(y*abs(np.cos(theta))/x)/np.pi; 
        print pedestrian,' passes threshold'
    return mesh_area*F/sa_ped
    
def wallviewfactors(pedestrian, keys, faces, mesh_area):
    print "***Generating wall view factors at", pedestrian
    visdic = {}
    for key in keys:
        try: 
            line = envuo.py3dmodel.construct.make_edge(pedestrian, key)
            intercept = []; append = intercept.append
            for face in faces: 
                if not envuo.py3dmodel.calculate.point_in_face(key,face): append(envuo.py3dmodel.calculate.intersect_edge_with_face(line,face))
                else: keyface = face
            if not sum(intercept,[]): 
                visdic.update({key:calc_wvf(pedestrian, key, keyface, mesh_area)}) 
        except (RuntimeError, UnboundLocalError): print "Not this one"
    return visdic


#def get_SET(model,SurfaceTemperatures,AirTemperatures,Pressure,Velocity,time_str,latitude,longitude)

def meanradtemp(pedkeys, allvisibility, shadowdic, svf, T_surf, R_surf, Ta):
    vp = 50*6.1121*np.exp((18.678-(Ta-273.2)/234.4)*(Ta-273.2)/(Ta-273.2+257.14))/1000
    E_sky = (Ta**4)*(0.82-0.25*10**(-0.00945*vp))
    sky_emis= 1.72*(vp/Ta)**(1/7.)
    Edf = pd.DataFrame(index=pedkeys, columns=('E_surf', 'E_r'))
    
    print 'Calculating surface radiation' 
    for ped in pedkeys:
        swapsurf = Ts.swaplevel(0, 2, axis=0)
        visibility = allvisibility[tuple(ped)]
        #visibility = pd.DataFrame.from_dict(visdic,orient='index')
        E_g = ground_emis*visibility*swapsurf.loc[swapsurf.index==0]**4 #Ts.loc[idx[:,:,[0]],:]
        E_w = wall_emis**visibility*swapsurf.loc[swapsurf.index>0]**4 
        E_r = visibility*Rs 
        Edf.loc[ped] = [sum(E_g.v)+sum(E_w.v),sum(E_r.v)]
        
    t_mrt=((shadowdic*E_sol*solarvf+E_dif*svf+sum(E_r.v))*(1-albedo)/sigma+sky_emis*E_sky*svf+sum(E_g.v))**(1/4.); 
    return t_mrt