# -*- coding: utf-8 -*-
"""
Created on Mon May 08 00:34:26 2017
Example of building model of idealized cubes from CSV file
NOTE: Building footprint must be square; height and streetwidth can be variable

@author: SHARED1-Tiffany
"""
import os
import pyliburo
import ExtraFunctions

origin = ExtraFunctions.quickline((0,0,0),3)
#==============METHOD #1: From a  CSV File =====================================
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
testfile = os.path.join(parent_path,'Examples','Topology.csv') # This is a CSV of 0 and height values that determine the location and size of each cube. The delimiter is a whitespace.

moddict1 = ExtraFunctions.makemodel_frmcsv(testfile,0.5,delimiter_str=' ') #the function saves a dictionary of the model and some other attributes. 

#Visualization

pyliburo.py3dmodel.construct.visualise([[origin]]+[[moddict1['model']]],['BLACK','BLUE'])
#%%
#==============METHOD #2: From dimensions =====================================
[street,width,height] = [5.,3.,4.]
moddict2 = ExtraFunctions.makemodelmatrix((3,5),street,width,height)

pyliburo.py3dmodel.construct.visualise([[origin]]+[[moddict2['model']]],['BLACK','RED'])

#%%
#================ SVf calculations=====================================
pedkey = (15.,8.,0.) #location of the pedestrian
pedline = ExtraFunctions.quickline(pedkey,1.5)

print 'SVF at ', pedkey, ' is ', pyliburo.skyviewfactor.calc_SVF(pedkey,moddict2['model'])

pyliburo.py3dmodel.construct.visualise([[pedline]]+[[moddict2['model']]],['BLACK','RED'])
