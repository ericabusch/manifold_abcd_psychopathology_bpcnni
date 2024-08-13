'''
plot_utils.py

various functions and settings for making the plots in `results.ipynb`

ELB 2024
'''

import numpy as np
import pandas as pd
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from config import * 
from stats_functions import extract_results

SEED=4

p = sns.color_palette("Set3_r",10)
symbols = {'0.1':'~', '0.05':'*', '0.01':'**', '0.001':'***'}

darkest_pink = '#A04248'
dark_pink = '#ED5F68'
medium_pink='#FEDBDB'
light_pink = '#FDB7B7'

darkest_green = '#07282A'
dark_green = '#0F5257'
medium_green = '#50A69C'
light_green = '#B2E1D9'
lightest='#FFFFFF'

embed_order = ['EPHATE_5','vanilla_PHATE','PCA','brain']

colors_2b0b = {'EPHATE_5':dark_green, 'vanilla_PHATE':medium_green, 'brain':lightest}
outline_2b0b = {i:dark_green for i in embed_order}

colors_evn = {'EPHATE_5':dark_pink, 'vanilla_PHATE':medium_pink, 'brain':lightest}
outline_evn = {i:dark_pink for i in embed_order}

hatch_dict = {'EPHATE_control':'\\', 'brain':'', "EPHATE_ADI":'oo', 
              'EPHATE_SSCEY':'--', 'EPHATE_5':'','PHATE_PLUS_FEATURES':'+','UMAP':'*'}

secondary_color_dict_evn = {'PCA':light_pink,'UMAP':medium_pink,'EPHATE_5':dark_pink}
secondary_color_dict_2b0b = {'PCA':light_green,'UMAP':medium_green,'EPHATE_5':dark_green}

# goes from lightest to darkest
zbtb_colors = ['#FFFFFF',light_green, dark_green] #'#2AA994','#115D62']
evn_colors = ['#FFFFFF',light_pink, dark_pink] #'#9C92A3','#4A2249']
colors_dict = {'brain':0, 'vanilla_PHATE':1, 'EPHATE_5':2, 'EPHATE_control':0, 'brain':0, 'EPHATE_ADI':3, 'EPHATE_SSCEY':4,'PHATE_PLUS_FEATURES':0,
              'PCA':0, 'UMAP':1}
big_colors = [darkest_pink, darkest_green]

evn_colors_hatch = ['#FFFFFF','#FFFFFF',dark_pink,'#FFFFFF','#FFFFFF',]
zbtb_colors_hatch = ['#FFFFFF','#FFFFFF',dark_green,'#FFFFFF','#FFFFFF']


xvals3 = {'brain_vanilla_PHATE': (-0.3, -0.02), 'EPHATE_5_brain': (-0.3, 0.3), 'EPHATE_5_vanilla_PHATE':(0.02, 0.3)}

xvals3 = {'brain_vanilla_PHATE': (-0.25, -0.05), 'EPHATE_5_brain': (-0.25, 0.25), 'EPHATE_5_vanilla_PHATE':(0.05, 0.25)}
yvals3 = {'brain_vanilla_PHATE':0.22, 'EPHATE_5_vanilla_PHATE':0.22, 'EPHATE_5_brain': 0.25}

xvals2 =  {'brain_vanilla_PHATE': (-0.25, 0.25)}
yvals2 = {'brain_vanilla_PHATE':0.55}
mids = [0, 1]

pal_evn = sns.color_palette([colors_evn[i] for i in colors_evn.keys()])
pal_2b0b = sns.color_palette([colors_2b0b[i] for i in colors_2b0b.keys()])

def barplot_nosignif(dataframe, yname, outfn, hue_label, hue_order=None, ylabel=None, xname='contrast', columns='ROI_name', 
                                ylim=None, yax=False):
    if not ylabel:
        ylabel = yname
    if columns=='ROI_name':
        column_order=REGIONS
    if not hue_label:
        hue_label = 'embedding_type'
        hue_order = method_order
    
    COLOR_IDX = sorted([colors_dict[h] for h in hue_order])
    X_LABELS = CONTRASTS#['emotion vs neutral','2-back vs 0-back']
    BIG_COLORS = {'nBack_emotion_vs_neutface':big_colors[0],
                 'nBack_2_back_vs_0_back':big_colors[-1]}
    
    sns.set(style='white')
    g=sns.catplot(data=dataframe,  x=xname, y=yname, hue=hue_label, errorbar=('ci', 95), n_boot=1000, capsize=.05, seed=SEED,
                hue_order=hue_order, order=X_LABELS, kind='bar', col=columns, col_order=column_order,err_kws={'linewidth': 1.5,'color': 'k'})
    
    g.fig.subplots_adjust(top=0.84) 
    g.fig.suptitle(f'prediction of n-Back performance')
    g._legend.remove()
    for ax in g.axes[0]:
        if yax: ax.axhline(0, color='k', linestyle='--')
    
    if ylim != None and len(ylim)>2:  g.set(yticks=ylim)
    else: g.set(ylim=ylim)

    colors_in_order = []
    edge_colors = []
    embd_roi = []
    COLORMAPS = {'nBack_emotion_vs_neutface': [evn_colors[r] for r in COLOR_IDX],
                'nBack_2_back_vs_0_back': [zbtb_colors[r] for r in COLOR_IDX]}
    color_order = []
    edge_order = []
    for h in COLOR_IDX:
        sublist = []
        sublist2=[]
        for x in X_LABELS:
            sublist.append(COLORMAPS[x][h])
            sublist2.append(BIG_COLORS[x])
        color_order.append(sublist)
        edge_order.append(sublist2)
    

    for j, ax in enumerate(g.axes[0]):
        for bars, colors, edgecolors in zip(ax.containers, color_order,edge_order):
            for bar, color, edgec in zip(bars, colors, edgecolors):
                bar.set_facecolor(color)
                bar.set_edgecolor(edgec)

    if outfn is not None:
        plt.savefig(outfn, bbox_inches = "tight",transparent=True, format='pdf')
        plt.clf()
        
def barplot_signif(dataframe, stat_df, yname, outfn, hue_label, hue_order=None, ylabel=None, xname='contrast', columns='ROI_name', 
                                ylim=None, yax=False, symbol_type='symbol_pperm', xvals=xvals3, yvals=yvals3, xlabels=None, title='',hatch=None):
    if not ylabel:
        ylabel = yname
    if columns=='ROI_name':
        column_order=REGIONS
        column_names=REGION_NAMES
    else:
        column_order=None
        column_names=None
    if not hue_label:
        hue_label = 'embedding_type'
        hue_order = method_order
    
    COLOR_IDX = sorted([colors_dict[h] for h in hue_order])
    print(COLOR_IDX)
    X_LABELS = CONTRASTS
    BIG_COLORS = {'nBack_emotion_vs_neutface':big_colors[0],
                 'nBack_2_back_vs_0_back':big_colors[-1]}
    
    sns.set(style='white')
    g=sns.catplot(data=dataframe,  x=xname, y=yname, hue=hue_label, errorbar=('ci', 95), n_boot=1000, capsize=.05, seed=SEED,  
                hue_order=hue_order, order=X_LABELS, kind='bar', 
                col=columns, col_order=column_order, linewidth=2,alpha=0.8, err_kws={'linewidth': 2, 'color': 'k'},
                  hatch=hatch)
    
    g.fig.subplots_adjust(top=0.8) 
    g.fig.suptitle(title, fontsize=28)
    g._legend.remove()
    for ax in g.axes[0]:
        if yax: ax.axhline(0, color='k', linestyle='--')
    
    if ylim != None and len(ylim)>2:  
        g.set(yticks=ylim, ylabel=ylabel, fontsize=20)
    else:
        g.set(ylim=ylim, ylabel=ylabel)#; 
        g.set_ylabels(ylabel, fontsize=16)
    
    if xlabels: g.set_xticklabels(xlabels, fontsize=13) 

    colors_in_order = []
    edge_colors = []
    embd_roi = []
    COLORMAPS = {'nBack_emotion_vs_neutface': [evn_colors[r] for r in COLOR_IDX],
                'nBack_2_back_vs_0_back': [zbtb_colors[r] for r in COLOR_IDX]}
    color_order = []
    edge_order = []
    for h in COLOR_IDX:
        sublist = []
        sublist2=[]
        for x in X_LABELS:
            sublist.append(COLORMAPS[x][h])
            sublist2.append(BIG_COLORS[x])
        color_order.append(sublist)
        edge_order.append(sublist2)
    

    for j, ax in enumerate(g.axes[0]):
        for bars, colors, edgecolors in zip(ax.containers, color_order,edge_order):
            for bar, color, edgec in zip(bars, colors, edgecolors):
                bar.set_facecolor(color)
                bar.set_edgecolor(edgec)
                
    # now add on significance bars
    for i,a in enumerate(g.axes[0]):
        r = column_order[i]
        nr = column_names[i]
        
        for combostr in stat_df.combostr.unique():

            cols = ['combostr','ROI_name','contrast']
            vals = [combostr, r]
            first_str = extract_results(stat_df, cols, vals+[X_LABELS[0]], symbol_type).item()
            second_str = extract_results(stat_df, cols, vals+[X_LABELS[1]], symbol_type).item()

            xl = xvals[combostr]
            # if there's the thing, draw the line
            if first_str != None:
                x00 = xvals[combostr][0]+mids[0]
                x01= xvals[combostr][1]+mids[0]
                a.hlines(yvals[combostr], x00, x01,color='k')
                a.text(x=np.mean((x00, x01))-0.05, y=yvals[combostr], s=first_str, size=16)

            if second_str != None:
                x10 = xvals[combostr][0]+mids[1]
                x11= xvals[combostr][1]+mids[1]
                a.hlines(yvals[combostr], x10, x11,color='k')
                a.text(x=np.mean((x10, x11))-0.05, y=yvals[combostr], s=second_str, size=16)
        a.hlines(0, -0.5,1.5, color='k', linestyle='--')
        a.set_title(nr, fontsize=20)
        a.set(xlabel=None)

    if outfn is not None:
        plt.savefig(outfn, bbox_inches = "tight",transparent=True, format='pdf')
        plt.clf()
        
        

def barplot_signif_patterns(dataframe, stat_df, yname, outfn, hue_label, hue_order=None, ylabel=None, xname='contrast', columns='ROI_name', 
                                ylim=None, yax=False, symbol_type='symbol_pperm', xvals=xvals3, yvals=yvals3, xlabels=None, title=''):
    if not ylabel:
        ylabel = yname
    if columns=='ROI_name':
        column_order=REGIONS
        column_names=REGION_NAMES
    else:
        column_order=None
        column_names=None
    if not hue_label:
        hue_label = 'embedding_type'
        hue_order = method_order
    COLOR_IDX = [colors_dict[h] for h in hue_order]
    X_LABELS = CONTRASTS#['emotion vs neutral','2-back vs 0-back']
    BIG_COLORS = {'nBack_emotion_vs_neutface':big_colors[0],
                 'nBack_2_back_vs_0_back':big_colors[-1]}
    
    sns.set(style='white')
    g=sns.catplot(data=dataframe,  x=xname, y=yname, hue=hue_label, errorbar=('ci', 95), n_boot=1000, capsize=.05, seed=SEED, 
                hue_order=hue_order, order=X_LABELS, kind='bar', col=columns, col_order=column_order, linewidth=2, err_kws={'linewidth': 2, 'color': 'k'},alpha=0.8)
    
    g.fig.subplots_adjust(top=0.8) 
    g.fig.suptitle(title, fontsize=28)
    g._legend.remove()
    for ax in g.axes[0]:
        if yax: ax.axhline(0, color='k', linestyle='--')
    
    if ylim != None and len(ylim)>2:  
        g.set(yticks=ylim, ylabel=ylabel, fontsize=20)
    else:
        g.set(ylim=ylim, ylabel=ylabel)#; 
        g.set_ylabels(ylabel, fontsize=16)
    
    if xlabels: g.set_xticklabels(xlabels, fontsize=13) 

    colors_in_order = []
    edge_colors = []
    embd_roi = []
    COLORMAPS = {'nBack_emotion_vs_neutface': [evn_colors_hatch[r] for r in COLOR_IDX],
                'nBack_2_back_vs_0_back': [zbtb_colors_hatch[r] for r in COLOR_IDX]}
    
    HATCHMAPS = {'nBack_emotion_vs_neutface': [hatch_dict[h] for h in hue_order],
                'nBack_2_back_vs_0_back': [hatch_dict[h] for h in hue_order]}
    
    print(COLOR_IDX, COLORMAPS, HATCHMAPS)
    
    color_order = []
    edge_order = []
    hatch_order = []
    for i,h in enumerate(COLOR_IDX):
        sublist = []
        sublist2=[]
        sublist3=[]
        for x in X_LABELS:
            sublist.append(COLORMAPS[x][i])
            sublist2.append(BIG_COLORS[x])
            sublist3.append(HATCHMAPS[x][i])
        color_order.append(sublist)
        edge_order.append(sublist2)
        hatch_order.append(sublist3)
    

    for j, ax in enumerate(g.axes[0]):
        for bars, colors, edgecolors, hatches in zip(ax.containers, color_order,edge_order, hatch_order):
            for bar, color, edgec, hatch in zip(bars, colors, edgecolors, hatches):
                bar.set_facecolor(color)
                bar.set_edgecolor(edgec)
                bar.set_hatch(hatch)
                
    # now add on significance bars
    for i,a in enumerate(g.axes[0]):
        r = column_order[i]
        nr = column_names[i]
        
        for combostr in stat_df.combostr.unique():

            cols = ['combostr','ROI_name','contrast']
            vals = [combostr, r]
            first_str = extract_results(stat_df, cols, vals+[X_LABELS[0]], symbol_type).item()
            second_str = extract_results(stat_df, cols, vals+[X_LABELS[1]], symbol_type).item()

            xl = xvals[combostr]
            # if there's the thing, draw the line
            if first_str != None:
                x00 = xvals[combostr][0]+mids[0]
                x01= xvals[combostr][1]+mids[0]
                a.hlines(yvals[combostr], x00, x01,color='k')
                a.text(x=np.mean((x00, x01))-0.05, y=yvals[combostr], s=first_str, size=16)

            if second_str != None:
                x10 = xvals[combostr][0]+mids[1]
                x11= xvals[combostr][1]+mids[1]
                a.hlines(yvals[combostr], x10, x11,color='k')
                a.text(x=np.mean((x10, x11))-0.05, y=yvals[combostr], s=second_str, size=16)
        a.hlines(0, -0.5,1.5, color='k', linestyle='--')
        a.set_title(nr, fontsize=20)
        a.set(xlabel=None)

    if outfn is not None:
        plt.savefig(outfn, bbox_inches = "tight",transparent=True, format='pdf')
        plt.clf()