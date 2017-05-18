# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 06:38:46 2017
Output from Feb28  - no stagger
@author: Tiffany
"""
import time
import numpy
import scipy.io
import seaborn as sns #seaborn messes fonts up
import matplotlib.pyplot as plt
import pandas as pd
simdate = 'Mar19'

    #%% Importing TMRT and SET
current_path = os.path.dirname("__file__")
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
folderpath = os.path.join(parent_path, "StaggerStraight","Outputs_"+simdate)

#for config in cases:
    #config['TMRT'] = thermalcomfort.pdcoord(os.path.join(folderpath,'TMRT',config['name'] +'_'+ simdate+'_TMRT.csv'))
    #config['SVF'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name']  +'_' +simdate+ '_SVF.csv'))
    
#    config['Long'] = pdcoord(os.path.join(folderpath,config['name'] + '_Jan17_Long.csv'))
#    config['SET'] = thermalcomfort.pdcoord(os.path.join(folderpath,config['name']  +'_'+ simdate + '_SET.csv'))
#    
#    config['TMRT'].data.v= config['TMRT'].data.v-273.15
#    config['SET'].data.v= config['SET'].data.v-273.15
    
#
#A0['TMRT'].scatter3d()
#A0['SET'].scatter3d()


#%% 

#for (quant,name) in [('SVF','Sky View Factor'),('TMRT',u'Mean Radiant Temperature (C\N{DEGREE SIGN})'),('SET',u'Standard Effective Temperature (C\N{DEGREE SIGN})')]:
#    fig,ax = plt.subplots(1,1, figsize = (12,6))
#    plt.suptitle('Histogram of ' + quant + ' for different Lp ',size=15)
#    for config in cases:
#        ax.hist(config[quant].data.v, 50, normed=1, alpha=0.75, label = 'Lp = '+str(config['Lp']) ,edgecolor = "none")
#    ax.legend( loc=1,ncol=3)
#    ax.set_xlabel(name)
#    ax.set_ylabel('Counts')


#%%
#colors = ['b','g','r']
#for (quant,name) in [('TMRT',u'Mean Radiant Temperature (C\N{DEGREE SIGN})'),('SET',u'Standard Effective Temperature (C\N{DEGREE SIGN})'),('wind','Velocity (m/s)')]:
#    fig,ax = plt.subplots(1,1, figsize = (12,10))
#    plt.suptitle('Distribution of ' + quant + ' for different Lp ',size=20)
#    
#    
#    vplot = ax.violinplot([case[quant].data.v.() for case in cases],vert=0, widths=0.3, showmeans =1, showextrema=1)
#    for patch, color in zip(vplot['bodies'], colors): patch.set_color(color)
#
#    ax.legend( loc=1,ncol=3)
#    ax.set_yticks( [1, 2, 3])
#    ax.set_yticklabels(['Lp = 0.0625','Lp = 0.25','Lp = 0.44'])
#%%
for case in [A0]: case['TMRT'].contour(resolution = 2, model=case['lil_square'], title = r'$T_{mrt}$',cbartitle=u'TMRT (C\N{DEGREE SIGN})')
for case in [A0]: case['SET'].contour(resolution = 2, model=case['lil_square'], title = u'SET',cbartitle = u'SET (C\N{DEGREE SIGN})')
for case in [A0]: case['wind'].contour(resolution = 1, model=case['lil_square'], title = r'Wind',cbartitle = 'Wind Speed (m/s)', zmax = 1, zmin = 0, bar = True)
#($\lambda_{p}$=' + str(case['Lp'])+'
#%% seaborn
def violinplot(cases,quant, title='', xlabel='', ylabel=''):
    fig,ax = plt.subplots(figsize = (12,5))
    newpd = pd.DataFrame({str(case['Lp']):case['SET'].data.v for case in cases})
    newpd=newpd.where(abs(newpd)>=10e-5, None) 
    sns.violinplot(data=newpd, palette="Set1", inner="box", scale ="area", orient = "v",bw ='silverman', cut=0, linewidth=1)
    sns.despine(left=True)
    fig.suptitle(title, fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel,size = 16,alpha=0.7)
    ax.set_xlabel(xlabel,size = 16,alpha=0.7)
    
#violinplot(cases,'SET')
#%%
bbox_props = dict(boxstyle="round,pad=0.1", fc="white", lw=0)

quant = 'SET'; xtitle = u'SET (C\N{DEGREE SIGN})'
fig,ax = plt.subplots(figsize = (7,7))
newpd = pd.DataFrame({str(case['Lp']):case[quant].data.v for case in cases})
newpd=newpd.where(abs(newpd)>=10e-5, None) 
sns.violinplot(data=newpd, palette="Set1", inner="box", scale ="area", orient = "v",bw =0.1, cut=0, linewidth=1)
sns.despine(left=True)
fig.suptitle('Distribution of ' + quant, fontsize=18, fontweight='bold')
ax.set_xlabel(r"Urban Density - $\lambda_{p}$",size = 16,alpha=0.7)
ax.set_ylabel(xtitle,size = 16,alpha=0.7)
fig.text(0.21, 0.85, r"median="+"%0.1f" % (newpd['0.0625'].median()), ha ='left', fontsize = 11,bbox=bbox_props)
fig.text(0.467, 0.85, r"median="+"%0.1f" % (newpd['0.25'].median()), ha ='left', fontsize = 11,bbox=bbox_props)
fig.text(0.725, 0.85, r"median="+"%0.1f" % (newpd['0.44'].median()), ha ='left', fontsize =11,bbox=bbox_props)

quant = 'TMRT'; xtitle = u'Celsius(C\N{DEGREE SIGN})'
fig,ax = plt.subplots(figsize = (7,7))
newpd = pd.DataFrame({str(case['Lp']):case[quant].data.v for case in cases})
newpd=newpd.where(abs(newpd)>=10e-5, None) 
sns.violinplot(data=newpd, palette="Set1", inner="box", scale ="area", orient = "v",bw =0.1, cut=0, linewidth=1)
sns.despine(left=True)
fig.suptitle(r'Distribution of $T_{mrt}$', fontsize=18, fontweight='bold')
ax.set_xlabel(r"Urban Density - $\lambda_{p}$",size = 16,alpha=0.7)
ax.set_ylabel(xtitle,size = 16,alpha=0.7)
fig.text(0.21, 0.85, r"median="+"%0.1f" % (newpd['0.0625'].median()), ha ='left', fontsize = 11,bbox=bbox_props)
fig.text(0.467, 0.85, r"median="+"%0.1f" % (newpd['0.25'].median()), ha ='left', fontsize = 11,bbox=bbox_props)
fig.text(0.725, 0.85, r"median="+"%0.1f" % (newpd['0.44'].median()), ha ='left', fontsize =11,bbox=bbox_props)


#%%
quant = 'TMRT_Ts'; xtitle = u'Celsius(C\N{DEGREE SIGN})'
fig,ax = plt.subplots(figsize = (7,7))
newpd2 = pd.DataFrame({str(case['Lp']):case[quant].data.v for case in cases})
newpd2=newpd2.where(abs(newpd2)>=10e-5, None) 
sns.violinplot(data=newpd2, palette="pastel", inner="box", scale ="area", orient = "v",bw =0.2, cut=0, linewidth=1.5,scale_hue =True)
sns.despine(left=True)
fig.suptitle(r'Distribution of $T_{mrt}$ with constant $T_{surf}$', fontsize=18, fontweight='bold')
ax.set_ylabel(r"Urban Density - $\lambda_{p}$",size = 16,alpha=0.7)
ax.set_xlabel(xtitle,size = 16,alpha=0.7)
#%%
xtitle = u'Celsius(C\N{DEGREE SIGN})'
fig,ax = plt.subplots(figsize = (9,9))
sns.set(font_scale=1.5)
#combined_pd = pd.concat([newpd, newpd2], axis=1, join_axes=[newpd.index])
temp1 = sum([[(case['Lp'],'Detailed_Ts',i) for i in case['TMRT'].data.v.ravel()] for case in cases],[])
temp2 = sum([[(case['Lp'],'Uniform_Ts',i) for i in case['TMRT_Ts'].data.v.ravel()] for case in cases],[]);
combined_pd = pd.DataFrame(temp1+temp2); 
combined_pd.columns =['Lp','Surf','v']
ax = sns.violinplot(x="Lp", y = "v", hue ="Surf", data = combined_pd,palette="muted",  inner="box", scale ="area" ,bw =0.1, cut=0, linewidth=1.5,scale_hue=False)
sns.despine(left=True)
fig.suptitle(r'Distribution of $T_{mrt}$', fontsize=18, fontweight='bold')
ax.set_xlabel(r"Urban Density - $\lambda_{p}$",size = 16,alpha=0.7)
ax.set_ylabel(xtitle,size = 16,alpha=0.7)
fig.text(0.21, 0.53, r"$\Delta_{mean} =$"+"%0.1f" % (combined_pd[combined_pd['Surf']=='Uniform_Ts'][combined_pd['Lp']==0.0625].v.mean() - combined_pd[combined_pd['Surf']=='Detailed_Ts'][combined_pd['Lp']==0.0625].v.mean()), ha ='left', fontsize = 11)
fig.text(0.467, 0.53, r"$\Delta_{mean} =$"+"%0.1f" % (combined_pd[combined_pd['Surf']=='Uniform_Ts'][combined_pd['Lp']==0.25].v.mean() - combined_pd[combined_pd['Surf']=='Detailed_Ts'][combined_pd['Lp']==0.25].v.mean()), ha ='left', fontsize = 11)
fig.text(0.725, 0.53, r"$\Delta_{mean} =$"+"%0.1f" % (combined_pd[combined_pd['Surf']=='Uniform_Ts'][combined_pd['Lp']==0.44].v.mean() - combined_pd[combined_pd['Surf']=='Detailed_Ts'][combined_pd['Lp']==0.44].v.mean()), ha ='left', fontsize =11)

#%%
cases=[A0,A1,A2]
quant = 'TMRT_second'; xtitle = u'Celsius(C\N{DEGREE SIGN})'
fig,ax = plt.subplots(figsize = (7,7))
newpd = pd.DataFrame({str(case['Lp']):case[quant].data.v for case in cases})
newpd=newpd.where(abs(newpd)>=10e-5, None) 
sns.violinplot(data=newpd, palette="Set1", inner="box", scale ="area", orient = "v",bw =0.1, cut=0, linewidth=1)
sns.despine(left=True)
fig.suptitle(r'Distribution of $T_{mrt}$', fontsize=18, fontweight='bold')
ax.set_ylabel(r"Urban Density - $\lambda_{p}$",size = 16,alpha=0.7)
ax.set_xlabel(xtitle,size = 16,alpha=0.7)
#%%
quant = 'Refl_wall'; xtitle = u'Reflect Irradiance [$W/m^{2}$]'
sns.set(font_scale=2.2)
fig,ax = plt.subplots(figsize = (7,8))
newpd2 = pd.DataFrame({str(case['Lp']):case[quant].data.v for case in cases})
newpd2=newpd2.where(abs(newpd2)>=10e-5, None) 
sns.violinplot(data=newpd2, palette="Reds_d", inner="box", scale ="area", orient = "v",bw =0.1, cut=0, linewidth=1.5,scale_hue =True)
sns.despine(left=True)
fig.suptitle(r'Distribution of Reflect Irradiance on Walls', fontsize=24, fontweight='bold')
ax.set_xlabel(r"Urban Density - $\lambda_{p}$",size = 24,alpha=0.7)
ax.set_ylabel(xtitle,size = 24,alpha=0.7)

quant = 'Tsurf_wall'; xtitle = u'Celsius(C\N{DEGREE SIGN})'
fig,ax = plt.subplots(figsize = (7,8))
newpd2 = pd.DataFrame({str(case['Lp']):case[quant].data.v-273.15 for case in cases})
newpd2=newpd2.where(abs(newpd2)>=10e-5, None) 
sns.violinplot(data=newpd2, palette="Blues_d", inner="box", scale ="area", orient = "v",bw =0.1, cut=0, linewidth=1.5,scale_hue =False)
sns.despine(left=True)
fig.suptitle(r'Distribution of $T_{surf}$ on Walls', fontsize=24, fontweight='bold')
ax.set_xlabel(r"Urban Density - $\lambda_{p}$",size = 24,alpha=0.7)
ax.set_ylabel(xtitle,size = 24,alpha=0.7)

#%%

bbox_props = dict(boxstyle="round,pad=0.1", fc="white", lw=0)

quant = 'wind'; xtitle = u'Wind speed (m/s)'
fig,ax = plt.subplots(figsize = (7,7))
newpd = pd.DataFrame({str(case['Lp']):abs(case[quant].data.v) for case in cases})
newpd=newpd.where(abs(newpd)>=10e-5, None) 
sns.violinplot(data=newpd, palette="Set1", inner="box", scale ="area", orient = "v",bw =0.1, cut=0, linewidth=1)
sns.despine(left=True)
fig.suptitle('Distribution of ' + quant, fontsize=18, fontweight='bold')
ax.set_xlabel(r"Urban Density - $\lambda_{p}$",size = 16,alpha=0.7)
ax.set_ylabel(xtitle,size = 16,alpha=0.7)
fig.text(0.21, 0.85, r"mean="+"%0.2f" % (newpd['0.0625'].mean()), ha ='left', fontsize = 11,bbox=bbox_props)
fig.text(0.467, 0.85, r"mean="+"%0.2f" % (newpd['0.25'].mean()), ha ='left', fontsize = 11,bbox=bbox_props)
fig.text(0.725, 0.85, r"mean="+"%0.2f" % (newpd['0.44'].mean()), ha ='left', fontsize =11,bbox=bbox_props)


#%% Albedo and Tmrt, SET
bbox_props = dict(boxstyle="round,pad=0.1", fc="white", lw=0)
sns.set(font_scale=1.5)
albedo_comparison = pd.DataFrame(sum([[(case['Lp'],case['albedo'],i) for i in case['SET'].data.v.ravel()] for case in cases],[])); 
albedo_comparison.columns = ['Lp','albedo','v']
albedo_comparison=albedo_comparison.where(abs(albedo_comparison)>=10e-5, None) 
albedo_comparison = albedo_comparison.convert_objects(convert_numeric=True)

ax = sns.factorplot(x="Lp", y = "v", hue ="albedo", data = albedo_comparison,size=6,palette="muted", kind="violin",bw =0.1, cut=0, linewidth=1.1,scale_hue =False,legend_out=False)
sns.set_style("whitegrid")
sns.plt.title(r'Effects of Wall Albedo on SET')
sns.plt.xlabel(r"Urban Density - $\lambda_{p}$",alpha=0.7)
sns.plt.ylabel( u'$SET$ ($C\degree$)',alpha=0.7) # Reflect Irradiance [$W/m^{2}$]
#sns.plt.ylabel( u'Reflect Irradiance [$W/m^{2}$]',alpha=0.7)
sns.plt.text(-0.3, 21.5, r"$\Delta_{median} =$"+"%0.1f" % (albedo_comparison[albedo_comparison['albedo']==0.3][albedo_comparison['Lp']==0.0625].v.median() - albedo_comparison[albedo_comparison['albedo']==0.1][albedo_comparison['Lp']==0.0625].v.median()), ha ='left', fontsize = 11,bbox=bbox_props)
sns.plt.text(0.7, 21.5, r"$\Delta_{median} =$"+"%0.1f" % (albedo_comparison[albedo_comparison['albedo']==0.3][albedo_comparison['Lp']==0.25].v.median() - albedo_comparison[albedo_comparison['albedo']==0.1][albedo_comparison['Lp']==0.25].v.median()), ha ='left', fontsize = 11,bbox=bbox_props)
sns.plt.text(1.7, 21.5, r"$\Delta_{median} =$"+"%0.1f" % (albedo_comparison[albedo_comparison['albedo']==0.3][albedo_comparison['Lp']==0.44].v.median() - albedo_comparison[albedo_comparison['albedo']==0.1][albedo_comparison['Lp']==0.44].v.median()), ha ='left', fontsize = 11,bbox=bbox_props)



