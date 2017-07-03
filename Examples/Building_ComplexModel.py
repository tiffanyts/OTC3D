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

#Step 2: load the shapefile into a list.
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
example_parent_path = os.path.join(parent_path,"Examples","Input_Data", "OSM_HDB_shapefiles")
shpfile1 = os.path.join(example_parent_path,"rivervale116.shp")
shpfile_list = [shpfile1]

#Step 2: read every building footprint from each shapefile
building_list = []
for shpfile in shpfile_list:
    buildings = pyliburo.shp2citygml.get_buildings(shpfile)
    if buildings:
        building_list.extend(buildings)

#Step 3: extrude every footprint to a height. This is a building. Here, we use height 15. (this can be further coded to read height attributes, or manually changed per building)
display_list = []
for block in buildings:
    face = block['geometry'][0]
    extrude = pyliburo.py3dmodel.construct.extrude(face, (0,0,1),15)
    display_list.append(extrude)
compound = pyliburo.py3dmodel.construct.make_compound([display_list[3]]+[display_list[7]])
boundingbox = pyliburo.py3dmodel.calculate.get_bounding_box(compound)
points1 = [(boundingbox[0],boundingbox[1],0),(boundingbox[0],boundingbox[4],0),(boundingbox[3],boundingbox[4],0),(boundingbox[3],boundingbox[1],0)]
groundface = pyliburo.py3dmodel.construct.make_polygon(points1) #makr your ground

#Step 4: Combine all buildings into a compound. This compound can be passed into the thermalcomfort module.
compound = pyliburo.py3dmodel.construct.make_compound([compound]+[groundface])

#Step 5:  Visualize it.
pyliburo.py3dmodel.construct.visualise([[compound]], ['BLUE'])

#%% PART 2 - Demo sky view factor. This is to help visualize the calculation.
# The calculation is complete using pyliburo.skyviewfactor.calc_SVF(coord,compound)

#Step 0: pick a spot for your pedestrian coordinates
(A,B,H) = [x+50 for x in boundingbox[0:2]]+[1.5]       
              
#Step 1:Build your 3D model. See above, we built "compound"
#Step 2: Calculate SVF
svf = pyliburo.skyviewfactor.calc_SVF((A,B,H),compound)
print "The sky view factor at ", A, ',',B ,',',H,' is ', svf

#Let's draw it out.
displayball = []; ballcolors=[];  
unitball = pyliburo.skyviewfactor.tgDirs(100)
visible=0; blocked = 0; r=20
#draw all the directions in the hemisphere with a radius of 20
for direction in unitball.getDirUpperHemisphere(): 
    (X,Y,Z) = (direction.x,direction.y,direction.z)
    displayball.append([pyliburo.py3dmodel.construct.make_edge((A,B,H),(r*X+A,r*Y+B,r*Z+H))])
    occ_interpt, occ_interface = pyliburo.py3dmodel.calculate.intersect_shape_with_ptdir(compound,(A,B,H),(X,Y,Z))
    # Color the directions accordingly
    if occ_interpt != None: ballcolors.append('RED'); blocked +=1. 
    else: ballcolors.append('GREEN'); visible +=1.
    
#how we calculated it:
#svf2 = (visible)/(visible+blocked)
#print 'Sky View Factor is', svf2

pedline = pyliburo.py3dmodel.construct.make_edge((A,B,0),(A,B,H))

pyliburo.py3dmodel.construct.visualise([[compound]] + [[pedline]] + displayball, ['BLUE']+['BLACK']+ ballcolors)


