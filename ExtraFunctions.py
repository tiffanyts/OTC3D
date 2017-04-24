# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:27:10 2016
THESE ARE FUNCTIONS TIFFANY USES TO:
Read Negin's output, visualize results, and build pythonOCC models (based on negin's model)
@author: SHARED1-Tiffany
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pyliburo
import numpy as np
import pandas as pd

def scatter3d_frompdcoord(pdcoord,title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d'); ax.pbaspect = [1, 1, 1] #always need pbaspect
    ax.set_title(title)
    p = ax.scatter(list(pdcoord.x), list(pdcoord.y),list(pdcoord.z),c = list(pdcoord.v),edgecolors='none', s=5) #*abs((Y-B)*mww)
    ax.view_init(elev=90, azim=-89)
    ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
    fig.colorbar(p)
    plt.draw()
    return fig
    
def scatter_pd_on_model(pdcoord,model,size=4):
    vertices = [(vertex.X(), vertex.Y(),vertex.Z()) for vertex in pyliburo.py3dmodel.fetch.vertex_list_2_point_list(pyliburo.py3dmodel.fetch.topos_frm_compound(model)["vertex"])]
    print max(vertices), min(vertices); 
    V1,V2,V3 = zip(*vertices);
    fig = plt.figure(); 
    ax = fig.add_subplot(111, projection='3d'); ax.pbaspect = [1, 1, 1] #always need pbaspect
    p = ax.plot_wireframe(V1,V2,V3)
    p = ax.scatter(list(pdcoord.x), list(pdcoord.y),list(pdcoord.z),c = list(pdcoord.v), s = size, edgecolors='none') 
    ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
    fig.colorbar(p)
    plt.draw()

def makeblock(botleft,w,hh): 
    """returns a block ready to be displayed """    
    points = []
    points.append(botleft)
    points.append([botleft[0],botleft[1]+w,botleft[2]])
    points.append([botleft[0]+w,botleft[1]+w,botleft[2]])
    points.append([botleft[0]+w,botleft[1],botleft[2]])
    tuplist = [tuple(x) for x in points]
    face = pyliburo.py3dmodel.construct.make_polygon(tuplist)
    cube = pyliburo.py3dmodel.construct.extrude(face, (0,0,1), hh)
    return cube

def makemodel_frmcsv(csv_matrix,delimiter_str,meshsize):
    """ Returns a display_list of OCC extruded squares based on a height-zero matrix from csv file  - no ground"""
    modmat = pd.read_csv(csv_matrix,delimiter=delimiter_str)
    dimx = modmat.shape
    modmat = modmat.astype(float)
    modmat = modmat.as_matrix()
    hei = float(modmat.max())
    bbc1 = np.unravel_index(modmat.argmax(),modmat.shape)

    bbc = [] ; display_list = [] ; width = []; street = []; streetstart = 0; #Cubes match up to topleft corners if input as rectangles. 
    for i in range(bbc1[0],dimx[0]-1):
        if (modmat[i,:] == modmat[i-1,:]).all() or (modmat[i,:] == 0).all():
            pass
        else:
            for j in range(bbc1[1],dimx[1]-1):
                if modmat[i,j] == modmat[i,j-1]:
                    pass
                elif modmat[i,j] < modmat[i,j-1]:
                    width.append(float(j - bbc[-1][1])*meshsize); streetstart = j
                    display_list.append(makeblock(tuple([val*meshsize for val in bbc[-1]]),width[-1],hei))
                else:               
                    bbc.append((i,j,0.))
                    street.append((j-streetstart)*meshsize)
    points1 = [(0,0,0), (0,dimx[1]*meshsize,0), (dimx[0]*meshsize,dimx[1]*meshsize,0),(dimx[0]*meshsize,0,0)]#clockwise
    groundface = pyliburo.py3dmodel.construct.make_polygon(points1)
    compound = pyliburo.py3dmodel.construct.make_compound(display_list)
    return ({"model":compound,"ground":groundface}, np.median(width), np.median(street), hei)
    
def makemodelmatrix((M,N),street,width,height):
    """ Returns a MxN matrix as a compound. """
    bbc = []; display_list = [];
#Build Ground
    dim = [street*d + width*d for d in (M,N)]
    (a,b,c) = (0,0,0) #origin
    points1 = [(a,b,c), (a,b+dim[0],c), (a+dim[1],b+dim[0],c),(a+dim[1],b,c)]#clockwise
    groundface = pyliburo.py3dmodel.construct.make_polygon(points1)
#Determine coordinates of top-left corners
    bbc.append([a+street/2.,b+street/2.,c])
    for i in range(0,N):
        for j in range (0,M):
            bbc.append([bbc[0][0]+i*(width+street),0,bbc[i-1][2]])
            bbc[-1][1] = bbc[0][1]+j*(width+street)
#Uses heights of cubes to build up matrix
    for i in range(1,len(bbc)):
        display_list.append(makeblock(bbc[i],width,height))#make all the cubes
    compound = pyliburo.py3dmodel.construct.make_compound(display_list)
    return {"model":compound,"ground":groundface}

def makemodel_frmshp(shpfile_list, def_height=5):
    building_list = []
    for shpfile in shpfile_list:
        buildings = pyliburo.shp2citygml.get_buildings(shpfile)
        if buildings:
            building_list.extend(buildings)
    display_list = []
    for block in buildings:
        bface = block['geometry'][0]
        try:
            extrude = pyliburo.py3dmodel.construct.extrude(bface, (0,0,1),float(block['height']))
        except (KeyError,RuntimeError):
            extrude = pyliburo.py3dmodel.construct.extrude(bface, (0,0,1),def_height)
        display_list.append(extrude)
    compound = pyliburo.py3dmodel.construct.make_compound(display_list)
    return compound


def makemodelstagger((M,N),street,width,height):
    """ Returns a MxN matrix as a compound. """
    bbc = []; display_list = [];
#Build Ground
    offset = (width+street)/2
    dim = [street*d + width*d + offset for d in (M,N)]

    (a,b,c) = (0,0,0) #origin
    points1 = [(a,b,c), (a,b+dim[0],c), (a+dim[1],b+dim[0],c),(a+dim[1],b,c)]#clockwise
    groundface = pyliburo.py3dmodel.construct.make_polygon(points1)
#Determine coordinates of top-left corners
    bbc.append([a+offset,b+offset,c])
    for i in range(0,N):
        for j in range (0,M):
            bbc.append([bbc[0][0]+i*(width+street),0,bbc[i-1][2]])
            bbc[-1][1] = bbc[0][1]+j*(width+street)- offset*(i%2)
#Uses heights of cubes to build up matrix
    for i in range(1,len(bbc)):
        display_list.append(makeblock(bbc[i],width,height))#make all the cubes
    compound = pyliburo.py3dmodel.construct.make_compound(display_list)
    return {"model":compound,"ground":groundface}
    
def make_sq_center(origin,x):
    a,b,c = origin
    square = pyliburo.py3dmodel.construct.make_polygon( [(a+x,b+x,c), (a+x,b-x,c), (a-x,b-x,c),(a-x,b+x,c)])
    return square

def quickline((X,Y,Z),Zend=0):
    return pyliburo.py3dmodel.construct.make_edge((X,Y,Z),(X,Y,Zend))
