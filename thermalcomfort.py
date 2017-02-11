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
import pyliburo
import pvlib
import fourpispace as fpi

from OCC.Display import OCCViewer
#from ExtraFunctions import *
Ndir = 250
unitball = fpi.tgDirs(Ndir)
sigma =5.    *10**(-8)

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

font = {'weight' : 'medium',
        'size'   : 22}

class pdcoord(object):
    """ Class for all x,y,z,v input files used in thermal comfort analysis. To initialize an empty pdcoord, put 0 instead of data """
    
    def __init__(self, csv_input, sep =','):
#        self.name = name
        self.data = read_pdcoord(csv_input,separator = sep)
    
    def recenter_to_origin(self):
        print 'shifted by x:' ,min(self.data.x), ' and y:',min(self.data.y)
        self.data['x'] = self.data['x'] - min(self.data.x)
        self.data['y'] = self.data['y'] - min(self.data.y)
        return self
    
    def repeat_clockwise(self,unit=1):
        """ extends pdcoord to quadrants 2,1,4 for repeating data"""
        shiftright = max(self.data.x) - min(self.data.x) + unit
        shiftdown = max(self.data.y) - min(self.data.y) + unit
        

        Q1 = pd.DataFrame(np.array([self.data.x+shiftright, 
                           self.data.y,
                           self.data.z, 
                           self.data.v]).T, 
                           columns = ['x','y','z','v'])
        Q2 = pd.DataFrame(np.array([self.data.x+shiftright, 
                           self.data.y-shiftdown,
                           self.data.z, 
                           self.data.v]).T, 
                           columns = ['x','y','z','v'])
        Q3 = pd.DataFrame(np.array([self.data.x, 
                           self.data.y-shiftdown,
                           self.data.z, 
                           self.data.v]).T, 
                           columns = ['x','y','z','v'])
        
        self.data =self.data.append(Q1,ignore_index=True)
        self.data =self.data.append(Q2,ignore_index=True)
        self.data =self.data.append(Q3,ignore_index=True)
        
        return self

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
        plt.rc('font', **font)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d'); ax.pbaspect = [1, 1, 1] #always need pbaspect
        ax.set_title(title)
        p = ax.scatter(list(self.data.x), list(self.data.y),list(self.data.z),c = list(self.data.v),edgecolors='none', s=size, marker = ",")
        ax.view_init(elev=90, azim=-89)
        ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
        fig.colorbar(p)
        
        plt.draw()
        try:
            vertices = [(vertex.X(), vertex.Y(),vertex.Z()) for vertex in pyliburo.py3dmodel.fetch.vertex_list_2_point_list(pyliburo.py3dmodel.fetch.topos_frm_compound(model)["vertex"])]
            V1,V2,V3 = zip(*vertices)            
            p = ax.plot_wireframe(V1,V2,V3 )

        except TypeError:
            pass        
        
        return fig
        
    def contour(self,title='',model=[], zmax = None, zmin = None, filename = None, lowres = 1):
        """ filename to save image. lowres to reduce resolution (high integer is lower resolution)"""
        plt.rc('font', **font)
        xi = np.linspace(min(self.data.x), max(self.data.x),len(set(self.data)))
        yi = np.linspace(min(self.data.y), max(self.data.y),len(set(self.data)))
        
    
#        zi = ml.griddata(self.data.x, self.data.y, self.data.v.interpolate(), xi, yi,interp='linear')
        
        #xi, yi = np.meshgrid(self.data.x, self.data.y)
        fig = plt.figure()
        plt.title(title)
        plt.contour(xi, yi, zi, 15, linewidths = 0, colors = 'k')
        plt.pcolormesh(xi, yi, zi, cmap = plt.get_cmap('rainbow'),vmax = zmax, vmin = zmin)
        plt.colorbar()        

        try:
            vertices = [(vertex.X(), vertex.Y()) for vertex in pyliburo.py3dmodel.fetch.vertex_list_2_point_list(pyliburo.py3dmodel.fetch.topos_frm_compound(model)["vertex"])]
            shape = patches.PathPatch(Path(vertices), facecolor='white', lw=0)
            plt.gca().add_patch(shape)
        except TypeError:
            pass
        
        try:
            fig.savefig(filename)
        except TypeError:
            return fig
    
#    def plot_along_line(self,X,Y, tick_list):
#        V = self.data.v
#        plt.plot(heights, SVFs_can, label='Canyon')
    
def pdcoords_from_pedkeys(pedkeys_np, values = np.zeros(0)):
    """ fills a pdcoord from numpy arrays """
    if not values.size:
        fill = [np.nan]*len(pedkeys_np)
    else: fill = values
    empty = pdcoord(zip(pedkeys_np.transpose()[0], pedkeys_np.transpose()[1], pedkeys_np.transpose()[2],fill))
    return empty
    
#%% Radiation Model Functions

"""
1) Calculate solar parameters
2) Calculate shadows, output pdcoord of shaded areas
For each pedestrian location:
3) Use fourpiradiation() to calculate sky view factor, list of visible intecept, and number of remaining directions that hit the ground (groundN)
4) Use call_values to call values of Ts, reflect, wall_albedo, wall_emissivity at intecepted locations on building surfaces
5) Use calc_radiation_from_values() to return longwave and shortwave radiation from surfaces
6) Use wall_emissivity*sigma*Ts**4*groundN/Ndir to calculate remaining ground longwave radiation
7) Use E_dif*solarvf + E_sol*solarvf*shadowint for direct and diffuse solar radiation
8) Tmrt = ((Eshort*(1-ped_albedo)+Elong)/sigma)**(1/4.)

 """
def solar_param((y,mo,d,h,mi),latitude,longitude, UTC_diff=0, groundalbedo=0.18):
    time_shift = datetime.timedelta(hours=UTC_diff) #SGT is UTC+8    
    thistime = pd.DatetimeIndex([pd.Timestamp(np.datetime64(datetime.datetime(y,mo,d,h,mi) + time_shift), tz='UTC')])  
    thisloc = pvlib.location.Location(latitude, longitude, tz='UTC', altitude=0, name=None)
    solpos = thisloc.get_solarposition(thistime)    
    
    sunpz = np.sin(np.radians(solpos.elevation[0])); hyp = np.cos(np.radians(solpos.elevation[0]))
    sunpy = hyp*np.cos(np.radians(solpos.azimuth[0]))
    sunpx = hyp*np.sin(np.radians(solpos.azimuth[0]))
    
    solar_pmt =  thisloc.get_clearsky(thistime,model='ineichen') #
    E_sol= solar_pmt.dni[0] #direct normal solar irradiation  [W/m^2]
    Ground_Diffuse = pvlib.irradiance.grounddiffuse(90,solar_pmt.ghi,albedo =groundalbedo)[0]  #Ground Reflected Solar Irradiation - vertical asphalt surface [W/m^2]
    Sky_Diffuse =  pvlib.irradiance.isotropic(90, solar_pmt.dhi)[0] #Diffuse Solar Irradiation - vertical surface[W/m^2].        
    
    #Formula 9 in Huang et. al. for a standing person, largely independent of gender, body shape and size. For a sitting person, approximately 0.25
    solarvf=abs(0.0355*np.sin(solpos.elevation[0])+2.33*np.cos(solpos.elevation[0])*(0.0213*np.cos(solpos.azimuth[0])**2+0.00919*np.sin(solpos.azimuth[0])**2)**(0.5)); 
    results = pd.DataFrame({
    'solarvector':[(sunpx,sunpy,sunpz)],
    'solarviewfactor':[solarvf],
    'direct_sol':[E_sol],
    'diffuse_frm_sky':[Sky_Diffuse],
    'diffuse_frm_ground':[Ground_Diffuse]
    })    
    return results


def check_shadow(key, model, solarvector):
    occ_interpt, occ_interface = pyliburo.py3dmodel.calculate.intersect_shape_with_ptdir(model,key,solarvector)
    if occ_interpt != None: return 0
    else: return 1 

def get_shadow(pedestrian_keys, model,solar_vector):
    """ Returns a dataframe of shadowed (0) and sunlit (1) locations. Ignores points that are on the wall (treats them as not shadowed)  """
    shadow = pdcoord(zip(pedestrian_keys.transpose()[0], pedestrian_keys.transpose()[1], pedestrian_keys.transpose()[2],np.zeros(len(pedestrian_keys)) ))
    shadow.data['v'] = shadow.data.apply(lambda row: check_shadow((row['x'], row['y'], row['z']),model, solar_vector), axis=1)
    return shadow

def skyviewfactor(ped, model):
    """ This function is replaced by fourpiradiation, which combines the calculation with groundview and wall visibility """
    visible=0.; blocked = 0.;
    for direction in unitball.getDirUpperHemisphere():
        (X,Y,Z) = (direction.x,direction.y,direction.z)
        occ_interpt, occ_interface = pyliburo.py3dmodel.calculate.intersect_shape_with_ptdir(model,ped,(X,Y,Z))
        if occ_interpt != None: blocked +=1.0
        else: visible +=1.0
    svf = (visible)/(visible+blocked);
    return svf

def fourpiradiation(ped, model):
    """ returns SVF, number of ground points (N), and list of intercepts. 
    For uniform ground temperature, do not include ground surface in model. Longwave irradiance from ground can be calculated as emissivity*sigma*groundtemp**4*N/Ndir 
    If ground temperature is not uniform, include the ground in the model, and radiation will be calculated with the other surfaces. """ 
    sky=0.; ground = 0.; intercepts=[]
    for direction in unitball.getDirUpperHemisphere():
        (X,Y,Z) = (direction.x,direction.y,direction.z)
        occ_interpt, occ_interface = pyliburo.py3dmodel.calculate.intersect_shape_with_ptdir(model,ped,(X,Y,Z))
        if occ_interpt != None: intercepts.append([occ_interpt.X(), occ_interpt.Y(), occ_interpt.Z()])
        else: sky +=1.0
    for direction in unitball.getDirLowerHemisphere():
        (X,Y,Z) = (direction.x,direction.y,direction.z)
        occ_interpt, occ_interface = pyliburo.py3dmodel.calculate.intersect_shape_with_ptdir(model,ped,(X,Y,Z))
        if occ_interpt != None: 
            intercepts.append([occ_interpt.X(), occ_interpt.Y(), occ_interpt.Z()])
            ground += int(int(occ_interpt.Z())==0) #if ground is included in model 
        else: ground +=1.0
    SVF = (sky)/(len(unitball.getDirUpperHemisphere()));
    GVF = (ground)/(len(unitball.getDirLowerHemisphere()));
    return SVF, GVF, np.array(intercepts)

def call_values(intercepts, surfpdcoord, gridsize):
    """ Given a list of intercepts, a pdcoord of surface values, and the grid size, a list of values is returned """
    visibletemps = [surfpdcoord.val_at_coord(target,gridsize).v.mean() for target in intercepts]
    return visibletemps
    
def calc_radiation_from_values(SurfTemp, SurfReflect, SurfAlbedo, SurfEmissivity):
    """ List of values for visible surface parameters. returns long and shortwave radiative components. Assumes that lists are in order and of the same length"""
    longwave =  sum([emissivity*sigma*temp**4/Ndir for temp, emissivity in zip(SurfTemp,SurfEmissivity)])
    shortwave =  sum([albedo*reflect/Ndir for reflect, albedo in zip(SurfReflect, SurfAlbedo)])
    return longwave, shortwave

def calc_Esky_emis(Ta,RH):
    """ returns scalar of longwave radiation from the sky, that needs to be factored by SVF  """
    vp = RH*6.1121*np.exp((18.678-(Ta-273.2)/234.4)*(Ta-273.2)/(Ta-273.2+257.14))/1000
    skyemis = 1.24*(vp/Ta)**(1/7.)
    Esky = sigma*skyemis*(Ta**4)*(0.82-0.25*10**(-0.0945*vp))
    return Esky

def meanradtemp(Esky,Esurf, Eground,Ereflect, solarparam, SVF, GVF,  pedestrian_albedo, shadow=False):
    Eshort =  solarparam.diffuse_frm_sky[0]*SVF/2 + solarparam.diffuse_frm_ground[0]*GVF/2 + solarparam.direct_sol[0]*solarparam.solarviewfactor[0]*shadow+ Ereflect 
    Elong = Esky*SVF/2+Esurf+Eground
    t_mrt= ((Eshort*(1-pedestrian_albedo)+Elong)/sigma)**(1/4.)   
    return t_mrt

#def all_mrt(pedkey,compound,pdTa,solarparam,model_inputs):
#    """ Accepts dataframe of solar parameters, model inputs"""
#    Esky = calc_Esky_emis(pdTa.val_at_coord(pedkey).v, RH)
#    svf, gvf, intercepts = fourpiradiation(pedkey, compound) #interceptped        
#    shadowint = check_shadow(pedkey, compound,solarparam.solarvector[0])
#        
#    SurfTemp, SurfReflect, SurfAlbedo, SurfEmissivity = [[x]*len(intercepts) for x in [model_inputs.surftemp[0], solarparam.direct_sol[0], model_inputs.wall_albedo[0], model_inputs.wall_emissivity[0]]] #instead of call values
#    Elwall, Eswall = calc_radiation_from_values(SurfTemp, SurfReflect, SurfAlbedo, SurfEmissivity)
#    Eground = model_inputs.ground_emissivity[0]*sigma*gvf/2*model_inputs.groundtemp[0]**4
#    
#    TMRT =  meanradtemp(Esky,Elwall, Eground,Eswall, solarparam,svf,gvf, ped_constants.body_albedo[0], shadow=shadowint)
#    
#    results = pd.DataFrame({
#    'TMRT':[TMRT],
#    'SVF':[svf],
#    'Elwall':[Elwall],
#    'Eswall':[Eswall],
#    'Eground':[Eground]
#    })    
#    return results

def all_mrt(pedkey,compound,pdTa,pdReflect,pedTs,solarparam,model_inputs):
    """ Accepts dataframe of solar parameters, model inputs"""
    Esky = calc_Esky_emis(pdTa.val_at_coord(pedkey).v, RH)
    svf, gvf, intercepts = fourpiradiation(pedkey, compound) #interceptped        
    shadowint = check_shadow(pedkey, compound,solarparam.solarvector[0])
    
    call_values(intercepts, surfpdcoord, model_inputs.gridsize)
    SurfTemp, SurfReflect = [call_values(intercepts, surfpdcoord, gridsize) for x in [pedTs, pdReflect]]
    SurfAlbedo, SurfEmissivity =  [[x]*len(intercepts) for x in [model_inputs.wall_albedo[0], model_inputs.wall_emissivity[0]]] #instead of call values
    Elwall, Eswall = calc_radiation_from_values(SurfTemp, SurfReflect, SurfAlbedo, SurfEmissivity)
    Eground = model_inputs.ground_emissivity[0]*sigma*gvf/2*model_inputs.groundtemp[0]**4
    
    TMRT =  meanradtemp(Esky,Elwall, Eground,Eswall, solarparam,svf,gvf, ped_constants.body_albedo[0], shadow=shadowint)
    
    results = pd.DataFrame({
    'TMRT':[TMRT],
    'SVF':[svf],
    'Elwall':[Elwall],
    'Eswall':[Eswall],
    'Eground':[Eground]
    })    
    return results

#%% SET Calculations 
k=0.155; #unit conversion factor
pt=101.325;   #local atmosphere presssure in kPa

def calc_SET(microclimate,ped_constants,ped_properties):
    """
    Parameters
    ---------
    ped_properties:  DataFrame with columns 
    ped_constants: Properties of a typical standing person. Dataframe with columns       'eff_radiation_surface_area_ratio'
    microclimate: DataFrame with columns        'air_temperature','wind_speed','mean_radiant_temperature','mean_static_pressure'
    """
    dubois_area = 0.202*ped_constants['mass']**0.425*ped_constants['height']**0.725
    body_mu = ped_constants['work']/ped_constants['met']
    heat_produced = ped_constants['met']*(1-body_mu)
    ped_properties['T_skin'] = 35.7 - 0.032*heat_produced/dubois_area #Auliciems and Szokolay pg 19
    
    Ta = microclimate['T_air']
    microclimate['water_vapor_pressure'] = wpa = np.exp(20.386-5132/Ta)*.133322368
    H = ped_constants['met'] - ped_constants['work']  
    pssk=np.exp(20.386-5132/ped_properties['T_skin'])*.133322368 #water vapor pressure at skin; units in kPa
    ped_properties['T_clothing'] = tcl = ped_properties['T_skin'] \
        - 0.0275*(H)\
        - k*ped_constants['Rcl']*((H)-3.05*(5.73-0.007*(H)-wpa) \
        - 0.42*((H)-58.15) \
        - 0.0173*ped_constants['met']*(5.87-wpa) \
        - 0.0014*ped_constants['met']*(34-Ta)) #cloth temperature [K]
    ped_properties['Lewis_ratio'] = lr = 15.15*ped_properties['T_skin']/273.2 
    Recl=ped_constants['Rcl']*k/(lr*ped_constants['icl'])  
    
    # heat transfer coefficients and operative temperature, pressure
    hsc=8.6*(microclimate['wind_speed']**0.53)*(pt/101.33)**.55; #W/m**2*kPa corrected convective transfer coefficients for sensible heat, 0.15<v<1.5, for standing pedestrian
    he=lr*hsc;  #evaporate heat transfer coefficient
    hesp=he*(101.33/microclimate['mean_static_pressure'])**0.45; ##standard evaporate heat transfer coefficient, from Gagge,1986     
    
    hr=4*ped_constants['body_emis']*sigma*ped_constants['eff_radiation_SA_ratio']*((tcl+microclimate['mean_radiant_temperature'])/2.)**3;   #radiative heat transfer coefficient
    hz=hr+hsc;
    Ia=1./(hz*ped_constants['fcl']);   #intrinsic insulation of the air layer
    hp=1./(Ia+ped_constants['Rcl']*k);   #Sensible Heat Transfer Coefficient
    Rea=1/(lr*ped_constants['fcl']*hsc);
    hep=1/(Rea+Recl);    #insensible heat transfer coefficient
    hsp=hp+hr; #overall sensible heat transfer caefficient            
        
    #Operative temperature and pressure
    to=(hr*microclimate['mean_radiant_temperature']+hsc*Ta)/(hr+hsc) #operative temperature
    ttso=(hp/hsp)*(to-273.2)+(1-hp/hsp)*(ped_properties['T_skin']-273.2)  #standard operative temperature[C]
    ppso=(hep/hesp)*wpa+(1.-hsp/hesp)*pssk; #standard operative pressure
    
    func = lambda st : (ttso - st -(ped_properties['skin_wetness']*(hesp+hsc)/hsp)*(ppso-0.5*.133322368*np.exp(20.386-5132/( st+273.2))))
    try:
        s_set = fsolve(func,0)[0] +273.2
    except: s_set = np.nan
    return s_set

