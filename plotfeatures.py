# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:51:06 2021

@author: LeoBoeger
"""

# plot script
##############################################################################
### IMPORT ###################################################################
# standard
import os
import sys
sys.path.append(os.path.expanduser('~'))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import PIL

#third party
import matplotlib.pyplot as plt



##############################################################################
### PLOTS ####################################################################


### Diameter Normalised plot
def plotDiameter(org, cor, norm, base, title):
    fig, ax=plt.subplots(figsize=(100,50))
    ax.plot(range(len(org.iloc[:,0])), org.iloc[:,0],label= 'orginal')
    ax.plot(range(len(org.iloc[:,0])), cor.iloc[:,0], label = 'corrected')
    ax.plot(range(len(org.iloc[:,0])), norm.iloc[:,0], label ='normalised')
    ax.plot(range(len(org.iloc[:,0])), base.iloc[:,0], label = 'baseline')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    plt.title(title+' pupil diameter over time')
    fig.subplots_adjust(bottom=0.2)
    
    return fig



### 3D center movement
def plot3DCenter(x, y, z, title):
    fig = plt.figure(figsize=(100,50))
    ax = fig.add_subplot(121, projection='3d')
    width = depth = 1
    bottom = np.zeros_like(z)
    ax.bar3d(x=x, y=y, dx=width, dy=depth, z=bottom, dz=z) # the actual frequency is defined by the dz, the hight, z is the starting point of the bar
    XYmax = max(max(x),max(y))
    XYmin = min(min(x),min(y))
    ax.set_xlim(XYmin, XYmax)
    ax.set_ylim(XYmin,XYmax)
    ax.set_zlabel('frequency')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title(title+' eye frequency, median centered')
    
    return fig


### Velocity on radar
def plotVeloRadar( angle, velo, title):
    fig = plt.figure(figsize=(100,50))
    ax = fig.add_subplot(projection='polar',rasterized=True)
    radar = ax.scatter(angle['angle'], velo['velocity pxl/s'], c=velo['velocity pxl/s'], alpha=0.75)
    plt.title(title+" eye velocity and angle from base")
    
    note = "note: As the coordinate system of images is upside down, this\nplot is as well. I.e. movements towards 90deg are forward eye\nconvergences"
    fig.text(.12, -0.1, note, bbox=dict(boxstyle='square', fc='w', ec='r'))
    
    cb = plt.colorbar(radar,ax = [ax], location = 'left')
    cb.set_label("velocity in pxl/s from previous frame")
    
    return fig



### distance on radar
def plotDistRadar(angle, dist, title):
    fig = plt.figure(figsize=(100,50))
    ax = fig.add_subplot(projection='polar', rasterized=True)
    #ax.plot(LCbased['angle'], LCbased['eucl dist'], alpha=0.5)
    radar = ax.scatter(angle['angle'], angle['eucl dist'], c= dist['velocity pxl/s'], alpha=0.75)
    plt.title(title+" eye distance and angle from base\nwith color coded velocity")
    
    note = "note: As the coordinate system of images is upside down, this\nplot is as well. I.e. movements towards 90deg are forward eye\nconvergences"
    fig.text(.14, -0.1, note, bbox=dict(boxstyle='square', fc='w', ec='r'))
    
    cb = plt.colorbar(radar,ax = [ax], location = 'left')
    cb.set_label("velocity pxl/s")
    
    return fig

def plotCbasedRot(x, y, title, xaxon, xlabel, ylabel):
    fig = plt.figure(figsize=(60,60))
    ax = fig.add_subplot()  
    ax.scatter( x, y)
    #XYmax = max(np.max(x),np.max(y))+4
    #XYmin = min(np.min(x),np.min(y))-4
    ax.set_xlim(-16,36)
    ax.set_ylim(-16,36)
    plt.axhline(c='black')   
    plt.axvline(c='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.suptitle("Rotated Coordinates of the {t}\n    x axis is paralell to {xax}".format(t=title, xax=xaxon))
    ax.annotate('geometric median',
            xy=(0.2, 0.2), xycoords='data',
            xytext=(0.7, 0.7), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=70"))
    
    return fig
                 

    


### Kinematics comparison Velocity, Angle and Distance
def plotKinematics(kine, centered, ymovement, title, w_size, fps):
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].plot(range(len(kine)), kine['velocity pxl/s'])
    axs[0].set_title('velocity pxl/s')
    axs[0].set_ylabel('pxl')
    axs[1].plot(range(len(kine)), ymovement)
    #axs[1].plot(range(len(kine)), ymovement, linestyle='dashed', alpha=0.2, c=(0,0,0))
    axs[1].axhline(y= 0,color= 'tab:red') 
    axs[1].set_title('rotated x movement from geometric median')
    axs[1].set_ylabel('pxl')
    axs[2].plot(range(len(kine)), centered['eucl dist'], 'tab:green')
    axs[2].set_title('euclidean distance from geometric median')
    axs[2].set_ylabel('pxl')
    fig.suptitle("{} eye kinematics, smoothed with SMA of window {} frames, {} sec".format(title,w_size, format(w_size/fps, '.2f')))
    plt.xlabel('time (frames)')
    
    return fig



### threshold plot
"""
fig = plt.figure(figsize =(20,6))
ax = fig.add_subplot()
ax.scatter( range(len(LCbased)), LCkineFilt['velocity pxl/s'])
ax.plot( range(len(LCbased)), LCbasedFilt['eucl dist'])
ax.plot( range(len(LCbased)), LCbasedFilt['angle'])

#ax2.scatter( LCkineFilt['velocity pxl/s'], LCbasedFilt['angle'], color='tab:orange')
"""

### Correlation of ODR and weighted diameter
def plotCorr( a,b, title, labela, labelb, keeplabelstart, keeplabelend):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(a, b)
    ax.set_ylabel(labelb)
    ax.set_xlabel(labela)
    wordsa = ' '.join(labela.split(' ')[keeplabelstart: keeplabelend])
    wordsb = ' '.join(labelb.split(' ')[keeplabelstart: keeplabelend])
    
    fig.suptitle("Correlation of {a} and {b} for {t}".format(a=wordsa, b= wordsb, t=title))
    
    return fig

def write_pdf(fname, figures, out):
    images = []
    for fig in figures:
        fig.canvas.draw()
        fig_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        images.append(fig_img)
    images[0].save(fname, save_all=True, append_images=images[1:])
    
##############################################################################
### UNIVERSAL ################################################################

def plotOrg2Norm(org, norm, title, cor=False, base=False):
    fig, ax=plt.subplots(figsize=(100,50))
    ax.plot(range(len(org.iloc[:,0])), org.iloc[:,0],label= 'orginal')
    ax.plot(range(len(org.iloc[:,0])), norm.iloc[:,0], label ='normalised')
    if cor:
        ax.plot(range(len(org.iloc[:,0])), cor.iloc[:,0], label = 'corrected')
    if base:
        ax.plot(range(len(org.iloc[:,0])), base.iloc[:,0], label = 'baseline')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.title(title+' over time')
    fig.subplots_adjust(bottom=0.2)
    
    return fig

def plotScatNLine(org, norm, title, cor=False, base=False):
    fig, ax=plt.subplots(figsize=(100,50))
    ax.plot(range(len(org.iloc[:,0])), norm.iloc[:,0], label ='normalised')
    ax.plot(range(len(org.iloc[:,0])), org.iloc[:,0],'or', label= 'peaks')
    if cor:
        ax.plot(range(len(org.iloc[:,0])), cor.iloc[:,0], label = 'corrected')
    if base:
        ax.plot(range(len(org.iloc[:,0])), base.iloc[:,0], label = 'baseline')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.title(title+' over time')
    fig.subplots_adjust(bottom=0.2)
    
    return fig


def plotOverTime(df, col,xname, yname):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(len(df)), df[col])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.title(yname+' over time')
    
    return fig

def plot3InSpace(df, colsNlabel, unit, title):
    fig = plt.figure()
    ax = fig.add_subplot()
    for col in colsNlabel:
        ax.plot(df[col[0]], df[col[1]], label=col[2])
    ax.set_xlabel('x [{u}]'.format(u=unit))
    ax.set_ylabel('y [{u}]'.format(u=unit))
    ax.legend()
    plt.title(title+' over space')
    
    return fig

def plotPeakBouts(org, peaks, ons, offs, signal=False, title='#', baseline=False):
    fig, ax=plt.subplots(figsize=(100,50))
    ax.plot(range(len(org.iloc[:,0])), org.iloc[:,0], label ='original data')
    ax.plot(range(len(signal.iloc[:,0])), signal.iloc[:,0], label ='signal')
    
    ax.plot(range(len(peaks.iloc[:,0])), peaks.iloc[:,0],'or', label= 'peaks')
    ax.scatter(range(len(ons.iloc[:,0])), ons.iloc[:,0], label = 'on', marker=6, c='tab:green')
    ax.scatter(range(len(offs.iloc[:,0])), offs.iloc[:,0], label = 'off', marker=7, c='tab:orange')
    if isinstance(baseline, pd.DataFrame) or isinstance(baseline, np.ndarray):
        ax.plot(range(len(baseline.iloc[:,0])), baseline.iloc[:,0], label ='threshold')
    if isinstance(baseline, int) or isinstance(baseline, float):
        ax.axhline(baseline, label ='threshold', c='red')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.title(title+' over time')
    fig.subplots_adjust(bottom=0.2)
    
    return fig

