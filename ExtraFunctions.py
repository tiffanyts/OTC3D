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

def npscatter(np,title=''):
    X,Y,Z,V = np.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d'); ax.pbaspect = [1, 1, 1] #always need pbaspect
    ax.set_title(title)
    p = ax.scatter(X, Y,Z,c = V ,edgecolors='none', s=5) #*abs((Y-B)*mww)
    ax.view_init(elev=90, azim=-89)
    ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
    fig.colorbar(p)
    plt.draw()
    return fig

def scatter3d_fromdic(dic,title=''):
    X,Y,Z = zip(*dic.keys())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d'); ax.pbaspect = [1, 1, 1] #always need pbaspect
    ax.set_title(title)
    p = ax.scatter(X, Y,Z,c = dic.values(),edgecolors='none', s=5) #*abs((Y-B)*mww)
    ax.view_init(elev=90, azim=-89)
    ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
    fig.colorbar(p)
    plt.draw()
    return fig

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

def scatter_on_model(dic,model):
    vertices = [(vertex.X(), vertex.Y(),vertex.Z()) for vertex in pyliburo.py3dmodel.fetch.vertex_list_2_point_list(pyliburo.py3dmodel.fetch.topos_frm_compound(model)["vertex"])]
    print max(vertices), min(vertices)
    X,Y,Z = zip(*dic.keys()); V1,V2,V3 = zip(*vertices);
    fig = plt.figure(); 
    ax = fig.add_subplot(111, projection='3d'); ax.pbaspect = [1, 1, 1] #always need pbaspect
    p = ax.plot_wireframe(V1,V2,V3)
    p = ax.scatter(X, Y,Z, c=dic.values(), s = 3, edgecolors='none') #*abs((Y-B)*mww)
    ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
#    fig.colorbar(p)
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
    
def dcfdcoord(cfd_inputfile):
    """ Input FLUENT csv files into coordinate positions """
    cfd_input= pd.read_csv(cfd_inputfile,delim_whitespace = True,skiprows=[0],header=None)
    cfd_input.columns = ['x','y','z','value']
    
    d = dict([((x,y,z),i) for x,y,z,i in zip(cfd_input.x, cfd_input.y,cfd_input.z,cfd_input.value)])
    return d
    
def read_pdcoord(cfd_inputfile,separator = ','):
    """ Input FLUENT csv files into coordinate positions """
    cfd_input= pd.read_csv(cfd_inputfile,sep=separator,skiprows=[0],header=None,names = ['x','y','z','v']) # r"\s+" is whitespace
    cfd_input.sortlevel(axis=0,inplace=True,sort_remaining=True)

    return cfd_input

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

def repeat_surf(tdict, N, street, width):
    """ Given a dictionary, street and building width, returns the dictionary repeated for a NxN repeated matrix""" 
    dict2 = {}
    shift = [n*(street+width) for n in range(0,N)]
    for (X,Y,Z) in tdict.iterkeys():
        [dict2.update({(X+col,Y+row,Z):tdict[(X,Y,Z)]}) for col in shift for row in shift]
    return dict2

def quart_res(dic):
    X,Y,Z = map(lambda x: sorted(list(set(x))), zip(*dic)); coarse_dic = {}; 
    for i in range(1,len(X),2):
        new_y = []; values_x1= []; values_x2=[]; 
        for j in range(1,len(Y),2):
            new_y.append(np.mean([Y[j], Y[j-1]]))
            values_x1.append(np.mean([dic[(X[i-1], Y[j],Z[0])], dic[(X[i-1], Y[j-1],Z[0])]]))
            values_x2.append(np.mean([dic[(X[i], Y[j],Z[0])], dic[(X[i], Y[j-1],Z[0])]]))
        new_x = np.mean([X[i], X[i-1]])
        coarse_dic.update({(new_x,new_y[i],Z[0]):np.mean([values_x1[i],values_x2[i]]) for i in range(len(values_x1))})
    return coarse_dic

def shift_dxdy(dic, dx, dy):
    """ Given a dictionary, street and building width, returns the dictionary moved up and over""" 
    dict2 = {(X+dx,Y+dy,Z):dic[(X,Y,Z)] for (X,Y,Z) in dic.iterkeys()}
    return dict2
        
    
def shrink_to_block(dic, lower,upper):
    """ Given symmetric dictionary, shrinks dictionary to square defined by (lower,lower) and (upper, upper)""" 
    stuff =  {(X+1-lower,Y+1-lower,Z):dic[(X,Y,Z)] for (X,Y,Z) in dic if (lower<= Y<=upper and lower<= X<=upper)}
    return stuff
    
def centerblock(dic, street,width):
    Tab = repeat_surf(dic,3,street,width)
    new_dic = {key:value for key, value in Tab.iteritems() if (street+width) < key[0] < (street+width)*2 and (street+width) < key[1] < (street+width)*2}
    return new_dic