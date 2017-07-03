# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 22:19:41 2017
Extruding from shapefiles using Pyliburo and demonstrating the sky view factor calculation
@author: Tiffany Sin, with help from https://github.com/chenkianwee/pyliburo.
This does not use the thermalcomfort model, only Pyliburo.
"""

import os

#downloaded package

#non-built in packages
import pyliburo
#%% PARI 1 - EXTRUDING A SHAPE FILE

current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
example_parent_path = os.path.join(parent_path,"Examples","Input_Data", "OSM_HDB_shapefiles")
shpfile1 = os.path.join(example_parent_path,"rivervale116.shp")
shpfile_list = [shpfile1]

building_list = []
for shpfile in shpfile_list:
    buildings = pyliburo.shp2citygml.get_buildings(shpfile)
    if buildings:
        building_list.extend(buildings)

display_list = []
for block in buildings:
    face = block['geometry'][0]
    extrude = pyliburo.py3dmodel.construct.extrude(face, (0,0,1),15)
    display_list.append(extrude)
compound = pyliburo.py3dmodel.construct.make_compound([display_list[3]]+[display_list[7]])
boundingbox = pyliburo.py3dmodel.calculate.get_bounding_box(compound)
points1 = [(boundingbox[0],boundingbox[1],0),(boundingbox[0],boundingbox[4],0),(boundingbox[3],boundingbox[4],0),(boundingbox[3],boundingbox[1],0)]
groundface = pyliburo.py3dmodel.construct.make_polygon(points1)

compound = pyliburo.py3dmodel.construct.make_compound([compound]+[groundface])

pyliburo.py3dmodel.construct.visualise([[compound]], ['BLUE'])

#%% PART 2 - Demo sky view factor.
(A,B,H) = [x+50 for x in boundingbox[0:2]]+[1.5]
displayball = []; ballcolors=[];     
unitball = pyliburo.skyviewfactor.tgDirs(100)
visible=0; blocked = 0; r=20
for direction in unitball.getDirUpperHemisphere():
    (X,Y,Z) = (direction.x,direction.y,direction.z)
    displayball.append([pyliburo.py3dmodel.construct.make_edge((A,B,H),(r*X+A,r*Y+B,r*Z+H))])
    occ_interpt, occ_interface = pyliburo.py3dmodel.calculate.intersect_shape_with_ptdir(compound,(A,B,H),(X,Y,Z))
    if occ_interpt != None: ballcolors.append('RED'); blocked +=1.
    else: ballcolors.append('GREEN'); visible +=1.
    

svf = (visible)/(visible+blocked)
print 'Sky View Factor is', svf

pedline = pyliburo.py3dmodel.construct.make_edge((A,B,0),(A,B,H))

pyliburo.py3dmodel.construct.visualise([[compound]] + [[pedline]] + displayball, ['BLUE']+['BLACK']+ ballcolors)


