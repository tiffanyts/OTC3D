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
import matplotlib.mlab as ml
from matplotlib.path import Path
import matplotlib.patches as patches


import numpy as np
import pandas as pd
import envuo
import pvlib
import fourpispace as fpi
import csv

from pvlib import solarposition as sp
from pvlib import irradiance as ir
from pvlib import clearsky as csky
from OCC.Display import OCCViewer
#from ExtraFunctions import *
Ndir = 500
unitball = fpi.tgDirs(Ndir)
sigma =5.67*10**(-8)
#%% Data is handled as pandas dataframes with x, y, z, and value columns

def read_pdcoord(cfd_inputfile,separator = ','):
    """ Input files into coordinate positions """
    if cfd_inputfile == 'empty': 
        cfd_input = pd.DataFrame(columns = ['x','y','z','v'])
    else:
        try:     # if it's a csv file, read csv
            cfd_input= pd.read_csv(cfd_inputfile,sep=separator,skiprows=[0],header=None,names = ['x','y','z','v']) # r"\s+" is whitespace
        except IOError:   # If it's a numpy array, convert to pdframe
            try:
                cfd_input = pd.DataFrame(np.array(cfd_inputfile), columns = ['x','y','z','v'])
            except IOError:
                print "Incorrect Input: read_pdcoord requires the input of a 4-column csv file, numpy array, or pandas dataframe." 
                return None
    cfd_input.sortlevel(axis=0,inplace=True,sort_remaining=True)
    return cfd_input

class pdcoord(object):
    """ Class for all x,y,z,v input files used in thermal comfort analysis. To initialize an empty pdcoord, put 0 instead of data """
    def __init__(self, csv_input, sep =','):
#        self.name = name
        self.data = read_pdcoord(csv_input,separator = sep)

    def empty(self):
        self.data = pd.DataFrame(columns = ['x','y','z','v'])

    def val_at_coord(self,listcoord, radius = 0.):
        """ Enter coordinates as a list. If only X or only X,Y are given, returns selected dataframe. Range of selection can be widened with a radius.  """
        minx, miny, minz = map(lambda a: a-radius, listcoord) #make bins
        maxx, maxy, maxz = map(lambda a: a+radius, listcoord)
    
        if len(listcoord) == 1: return self.data[(self.data.x <=maxx) & (self.data.x >= minx)]
        elif len(listcoord) == 2: return self.data[(self.data.x <=maxx) & (self.data.x >= minx) & (self.data.y <=maxy) & (self.data.y >= miny) ]
        elif len(listcoord) == 3: 
            try: return self.data[(self.data.x <=maxx) & (self.data.x >= minx) & (self.data.y <=maxy) & (self.data.y >= miny) &(self.data.z <=maxz) & (self.data.z >= minz)]
            except ValueError: print "No entry matches that coordinate" 

    def scatter3d(self,title='',size=40,model=[]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d'); ax.pbaspect = [1, 1, 1] #always need pbaspect
        ax.set_title(title)
        p = ax.scatter(list(self.data.x), list(self.data.y),list(self.data.z),c = list(self.data.v),edgecolors='none', s=size, marker = ",") #*abs((Y-B)*mww)
        ax.view_init(elev=90, azim=-89)
        ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
        fig.colorbar(p)
        
        plt.draw()
        try:
            vertices = [(vertex.X(), vertex.Y(),vertex.Z()) for vertex in envuo.py3dmodel.fetch.vertex_list_2_point_list(envuo.py3dmodel.fetch.topos_frm_compound(model)["vertex"])]
            V1,V2,V3 = zip(*vertices)            
            p = ax.plot_wireframe(V1,V2,V3 )

        except TypeError:
            pass        
        
        return fig
        
    def contour(self,title='',model=[], zmax = None, zmin = None):
        #self.data.v.fillna(0) =         
        xi = np.linspace(min(self.data.x), max(self.data.x),len(self.data))
        yi = np.linspace(min(self.data.y), max(self.data.y),len(self.data))
        
        zi = ml.griddata(self.data.x, self.data.y, self.data.v.interpolate(), xi, yi,interp='linear')
        
        #xi, yi = np.meshgrid(self.data.x, self.data.y)
        fig =  plt.figure()
        plt.title(title)
        plt.contour(xi, yi, zi, 15, linewidths = 0, colors = 'k')
        plt.pcolormesh(xi, yi, zi, cmap = plt.get_cmap('rainbow'),vmax = zmax, vmin = zmin)
        plt.colorbar()        
        
        try:
            vertices = [(vertex.X(), vertex.Y()) for vertex in envuo.py3dmodel.fetch.vertex_list_2_point_list(envuo.py3dmodel.fetch.topos_frm_compound(model)["vertex"])]
            shape = patches.PathPatch(Path(vertices), facecolor='white', lw=2)
            plt.gca().add_patch(shape)
        except TypeError:
            pass
        return fig
        

    
#%% Radiation Model Functions

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
    
#def get_shadow(pedestrian_keys, model,solar_vector):
#    """ Returns a dictionary of shadowed (0) and sunlit (1) locations. Ignores points that are on the wall (treats them as not shadowed)  """
#    shadow = {}
#    for key in pedestrian_keys:
#        occ_interpt, occ_interface = envuo.py3dmodel.calculate.intersect_shape_with_ptdir(model,key,solar_vector)
#        if occ_interpt != None: shadow.update({key: 0}) 
#        else: shadow.update({key:1})
#    return shadow

def skyviewfactor(ped, model):
    visible=0.; blocked = 0.;
    for direction in unitball.getDirUpperHemisphere():
        (X,Y,Z) = (direction.x,direction.y,direction.z)
        occ_interpt, occ_interface = envuo.py3dmodel.calculate.intersect_shape_with_ptdir(model['model'],ped,(X,Y,Z))
        if occ_interpt != None: blocked +=1.0
        else: visible +=1.0
    svf = (visible)/(visible+blocked);
    return svf

def fourpiradiation(ped, model):
    """ returns SVF, number of ground points (N), and list of intercepts. 
    For uniform ground temperature, do not include ground surface in model. Longwave irradiance from ground can be calculated as emissivity*sigma*groundtemp**4*N/Ndir 
    If ground temperature is not uniform, include the ground in the model, and radiation will be calculated with the other surfaces. """ 
    visible=0.; ground = 0.; intercepts=[]
    for direction in unitball.getDirUpperHemisphere():
        (X,Y,Z) = (direction.x,direction.y,direction.z)
        occ_interpt, occ_interface = envuo.py3dmodel.calculate.intersect_shape_with_ptdir(model,ped,(X,Y,Z))
        if occ_interpt != None: intercepts.append([occ_interpt.X(), occ_interpt.Y(), occ_interpt.Z()])
        else: visible +=1.0
    for direction in unitball.getDirLowerHemisphere():
        (X,Y,Z) = (direction.x,direction.y,direction.z)
        occ_interpt, occ_interface = envuo.py3dmodel.calculate.intersect_shape_with_ptdir(model,ped,(X,Y,Z))
        if occ_interpt != None: intercepts.append([occ_interpt.X(), occ_interpt.Y(), occ_interpt.Z()])
        else: ground +=1.0
    svf = (visible)/(len(unitball.getDirUpperHemisphere()));
    return svf, ground, np.array(intercepts)

def call_values(intercepts, surfpdcoord, gridsize):
    """ Given a list of intercepts, a pdcoord of surface values, and the grid size, a list of values is returned """
    visibletemps = [surfpdcoord.val_at_coord(target,gridsize).v.mean() for target in intercepts]
    return visibletemps
    
def calc_radiation_from_intercepts(SurfTemp, SurfReflect, SurfAlbedo, SurfEmissivity):
    """ List of values for visible surface parameters. returns long and shortwave radiative components. Assumes that lists are in order and of the same length"""
    longwave =  sum([emissivity*sigma*temp**4/Ndir for temp, emissivity in zip(SurfTemp,SurfEmissivity)])
    shortwave =  sum([(albedo)*(reflect)/Ndir for reflect, albedo in zip(SurfReflect, SurfAlbedo)])
    return longwave, shortwave
    
#def call_emiss(intercepts, Ts, gridsize, emisscoord,E_sol,E_dif ):
#    """ Given a list of intercepts, a pdcoord of surface emissivities, and the grid size of the surface temperatures, a pdframe of values at intercepts is returned """
#    visibleemiss = [emisscoord.val_at_coord(target,gridsize).v.mean() for target in intercepts]
#    #vis_pd = pdcoord(zip(intercepts.transpose()[0], intercepts.transpose()[1], intercepts.transpose()[2], visibletemps))
#    longwave = sum([emissivity*sigma*Ts**4/Ndir for emissivity in visibleemiss]) #longwave
#    shortwave =  sum([(1-emissivity)*(E_sol+E_dif)/Ndir for emissivity in visibleemiss])
#    return longwave, shortwave

def thermal_radiation(ped, model, surf_data, gridsize):
    """ outputs total thermal radiation [W/m2] on a pedestrian at a single location """
    intercepts = alldirections(ped,model)
    vis_temps = call_temp(intercepts, surf_data,gridsize)
    radiation = sigma*vis_temps[vis_temps.v.notnull()].v**4/(len(vis_temps[vis_temps.v.notnull()]))
    return radiation.v.sum()
        
def meanradtemp(pedkeys, shadowdic, svf, T_surf, R_surf, Ta):
    vp = 50*6.1121*np.exp((18.678-(Ta-273.2)/234.4)*(Ta-273.2)/(Ta-273.2+257.14))/1000
    E_sky = (Ta**4)*(0.82-0.25*10**(-0.00945*vp))
    sky_emis= 1.72*(vp/Ta)**(1/7.)
    Edf = pd.DataFrame(index=pedkeys, columns=('E_surf', 'E_r'))
    Tmrt = pdcoord(zip(pedkeys.transpose()[0], pedkeys.transpose()[1], pedkeys.transpose()[2],np.zeros(len(pedkeys)) ))

    print 'Calculating surface radiation at ', ped 
    for ped in pedkeys:
        print 'Calculating surface radiation at ', tuple(ped) 
        svf, groundN, wallintercepts = fourpiradiation(pedkeys[i], compound)
        Eground = groundN*ground_emis*sigma*groundtemp**4/500

        
    t_mrt=((shadowdic*E_sol*solarvf+E_dif*svf+sum(E_r.v))*(1-albedo)/sigma+sky_emis*E_sky*svf+sum(E_g.v))**(1/4.); 
    return t_mrt

#%% SET Calculations 

#def get_SET(model,SurfaceTemperatures,AirTemperatures,Pressure,Velocity,time_str,latitude,longitude)