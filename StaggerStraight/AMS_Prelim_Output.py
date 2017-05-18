# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:37:19 2017
AMS Output 
@author: SHARED1-Tiffany
"""
import time
import numpy
import scipy.io
import matplotlib.pyplot as plt
import pyliburo
#import thermalcomfort
#import ExtraFunctions
#%%
cases= [A0,A1,A2]
simdate = 'Mar10'
for config in [A0,A1,A2]:
    config["model"] = ExtraFunctions.makemodelmatrix((5,3),config['canyon'],config['cube'],config['cube'])["model"]
    config["square"] = ExtraFunctions.make_sq_center(pyliburo.py3dmodel.calculate.get_centre_bbox(config['model']),(config["canyon"]+config["cube"])/2)

    
#for config in [S0,S1,S2]:
#    config["model"] = makemodelstagger((6,3),config['canyon'],config['cube'],config['cube'])["model"]
#    midpoint = ((config['canyon']+config['cube'])*1.5+config['cube']/2,(config['canyon']+config['cube'])*2+config['cube']/2,1.5)
#    
#    config["square"] = make_sq_center(midpoint,(config["canyon"]+config["cube"])/2)

pyliburo.py3dmodel.construct.visualise([[A0['model']]],['BLUE'])

    #%% Importing TMRT and SET

current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
folderpath = os.path.join(parent_path, "StaggerStraight","Outputs_"+simdate)

for config in cases:
    config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +'_'+  simdate+'_TMRT.csv'))
#    config['SVF'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_SVF.csv'))
    config['SET'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name'] +  '_'+simdate+'_SET.csv'))
    

##%%
#
#testx = scipy.io.loadmat(os.path.join(parent_path,"StaggerStraight",'VelocityAtPedLevel_LES_LpStaAlig.mat'))
#V = {}
#A0['wind_input'] = testx['s'][0][0][0]
#A1['wind_input']  = testx['s'][0][0][1]
#A2['wind_input']  = testx['s'][0][0][2]
#S0['wind_input']  = testx['s'][0][0][3]
#S1['wind_input'] = np.fliplr(testx['s'][0][0][6])
#S2['wind_input']  = testx['s'][0][0][8]
#
#for config in [A0,A1,A2]:
#    a,c = (min(config['TMRT'].data.x),min(config['TMRT'].data.y)),(max(config['TMRT'].data.x)+2,max(config['TMRT'].data.y)+2)
#    xi = np.arange(a[0],c[0],1)
#    yi = np.arange(a[1],c[1],1)
#
#    pedkeys= np.array([(xi[i],yi[j],1.5) for i in range(len(xi)) for j in range(len(yi))])  
#    config['wind'] = pdcoords_from_pedkeys(pedkeys,abs(config['wind_input'].flatten()))
#    config['wind'].data  = config['wind'].data[config['wind'].data.v !=0]
#    
#for config in [S0,S1,S2]:
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
#    
#    Ydim,Xdim = Zt.shape
#    xt = np.arange(BL[0],BL[0]+Xdim); yt = np.arange(BL[1]-Ydim/2,BL[1]+Ydim/2)
#    Xt,Yt = np.meshgrid(xt, yt)
#    
#    
#    windkeys= np.array(zip(Xt.flatten(),Yt.flatten(),[1.5]*len(Xt.flatten())))
#    wind = abs(Zt.flatten())
#
#    config['wind']  = pdcoords_from_pedkeys(windkeys,wind)
#    config['wind'].data  = config['wind'].data[config['wind'].data.v !=0]
##
#    print config['name']
#%% 

for (quant,name) in [('TMRT',u'Mean Radiant Temperature (C\N{DEGREE SIGN})'),('SET',u'Standard Effective Temperature (C\N{DEGREE SIGN})')]:
    fig,ax = plt.subplots(2,1, figsize = (9,9))
    plt.suptitle('Histogram of ' + quant + ' for different Lp ',size=20)

    ax[0].hist(A0[quant].data.v, 50, normed=1, alpha=0.75, label = 'Lp = 0.0625' ,edgecolor = "none")
    ax[0].hist(A1[quant].data.v, 50, normed=1, alpha=0.75, label = 'Lp = 0.25',edgecolor = "none")
    ax[0].hist(A2[quant].data.v, 50, normed=1, alpha=0.75, label = 'Lp = 0.44',edgecolor = "none")
    ax[0].legend( loc=1,ncol=3)
    ax[0].set_xlabel(name)
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Aligned Configurations')

#    ax[1].hist(S0[quant].data.v, 50, normed=1, alpha=0.75, label = 'Lp = 0.0625' ,edgecolor = "none")
#    ax[1].hist(S1[quant].data.v, 50, normed=1, alpha=0.75, label = 'Lp = 0.25',edgecolor = "none")
#    ax[1].hist(S2[quant].data.v, 50, normed=1, alpha=0.75, label = 'Lp = 0.44',edgecolor = "none")
#    ax[1].legend(loc=1,ncol=3)
#    ax[1].set_xlabel(name)
#    ax[1].set_ylabel('Counts')
#    ax[1].set_title('Staggered Configurations')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(top=.9)
#%%
colors = ['b','g','r']
for (quant,name) in [('SVF','Sky View Factor'),('TMRT',u'Mean Radiant Temperature (C\N{DEGREE SIGN})'),('SET',u'Standard Effective Temperature (C\N{DEGREE SIGN})'),('wind','Velocity (m/s)')]:
    fig,ax = plt.subplots(2,1, figsize = (9,9))
    plt.suptitle('Distribution of ' + quant + ' for different Lp ',size=20)

    vplot = ax[0].violinplot([A0[quant].data.v,A1[quant].data.v,A2[quant].data.v],vert=0)
    for patch, color in zip(vplot['bodies'], colors): patch.set_color(color)

    ax[0].legend( loc=1,ncol=3)
    ax[0].set_xlabel(name)
    ax[0].set_title('Aligned Configurations')
    ax[0].set_yticks( [1, 2, 3])
    ax[0].set_yticklabels(['Lp = 0.0625','Lp = 0.25','Lp = 0.44'])

    vplot = ax[1].violinplot([S0[quant].data.v,S1[quant].data.v,S2[quant].data.v],vert=0)
    for patch, color in zip(vplot['bodies'], colors): patch.set_color(color)
    ax[1].legend(loc=1,ncol=3)
    ax[1].set_xlabel(name)
    ax[1].set_title('Staggered Configurations')
    ax[1].set_yticks( [1, 2, 3])
    ax[1].set_yticklabels(['Lp = 0.0625','Lp = 0.25','Lp = 0.44'])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(top=.9)

