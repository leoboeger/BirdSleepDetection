# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:01:56 2021

@author: LeoBoeger
"""
##############################################################################
### IMPORT ###################################################################
import os
import sys
sys.path.append(os.path.expanduser('~'))
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
from tqdm import trange

from pylibLeo import GeoMedian as gmL
#from pylibLeo import Smoothing as smL
#from pylibLeo import Normalising as nrL


##############################################################################
### SETTINGS #################################################################
"""
dirIn = 'C:/Users/labgrprattenborg/DeepLabCut/DLCprojects/test_videos/fMRIsleepBudapestResNet101Contrast-Leo-2021-04-20/iteration 0_1'
csvFile = 'Budapest_fMRI_contrasted2_1DLC_resnet_101_fMRIsleepBudapestResNet101ContrastApr14shuffle1_400000.csv'
#csvFile = 'Blue69_S2_Video_2021-01-12_09-30-00-000DLC_resnet_101_fMRIsleepBudapestResNet101ContrastApr14shuffle1_400000.csv'
video = "Budapest_fMRI_contrasted2.avi"
#video = "Blue69_S2_Video_2021-01-12_09-30-00-000_output.avi"
fps = 30

csv_dir_in = os.path.join(dirIn, csvFile)
csv_in = pd.read_csv(csv_dir_in, header=[1,2])

bp_names = list(csv_in.columns)

beak_col = []
for col in bp_names:
    for row in col:
        if "beak" in row:
            beak_col.append(col)
            break
BeakDf = csv_in[beak_col]

"""

##############################################################################
### FUNCTIONS ################################################################

### Gap ######################################################################
### Calculation of the euclidean distance for beak gap
def EuclDist(df, A, B, C):
    dist_beak = np.sqrt((df[(A, 'x')]-df[(B, 'x')])**2+
                       (df[(A, 'y')]-df[(B, 'y')])**2)
    lower = np.sqrt((df[(B, 'x')]-df[(C, 'x')])**2+
                       (df[(B, 'y')]-df[(C, 'y')])**2)
    upper = np.sqrt((df[(A, 'x')]-df[(C, 'x')])**2+
                       (df[(A, 'y')]-df[(C, 'y')])**2)
    
    eucl_df = pd.DataFrame(columns=['beak gap', 'lower side', 'upper side'])
    eucl_df['beak gap'] = dist_beak
    eucl_df['lower side'] = lower
    eucl_df['upper side'] = upper
    
    return eucl_df


### beak angle ################################################################
### Calculation of the angle of upper to lower over beakcorner

def BeakAngle(df, ab, bc, ca, mode='gamma'):
    angle = []
    if mode == 'gamma':
        for frame in trange(len(df)):
            gamma = np.arccos((df.iloc[frame][bc] + df.iloc[frame][ca] - df.iloc[frame][ab])
                           / (2 * df.iloc[frame][bc] * df.iloc[frame][ca]))
            angle.append(gamma)
    df['beakcorner angle'] = angle
    return


### Base position ############################################################
### Geometric median of Center

def beakCentered(df, bp2center):
    bps = df.columns.get_level_values(0).drop_duplicates().to_list()
    center_arr = df[bp2center][["x", "y"]].values
    base = gmL.geometric_median(center_arr)
    
    ### Coordinates of all frames transforemd on base
    col = pd.MultiIndex.from_product([bps,['x','y','angle','eucl dist']])
    c_based = pd.DataFrame(columns=col)
    for bp in bps:
        
        c_based[bp,'x'] = df[bp, 'x']-base[0]
        c_based[bp,'y'] = df[bp, 'y']-base[1]
        c_based[bp, 'eucl dist'] = np.sqrt(c_based[bp, 'x']**2+c_based[bp, 'y']**2)

    return base, c_based


### Velocity #################################################################
### euclidean distance to previous frame

def GapKinematics(df, dist2calc, fps):
    kine_df = pd.DataFrame()
    kine_df[dist2calc] = df[dist2calc]
    kine_df["dist prev"] = kine_df[dist2calc].shift(1)
    kine_df['diffr dist prev'] = kine_df[dist2calc]-kine_df["dist prev"]
    kine_df['velocity pxl/s'] = kine_df['diffr dist prev']*(1000/fps)
    kine_df['velocity pxl/s'] = kine_df['velocity pxl/s'].abs()

    return kine_df





"""
##############################################################################
### CALL F ###################################################################
# beak distance of all 3 bp´s + angle oover beak corner
BeakDist = EuclDist(BeakDf,  'beaktip_top', 'beaktip_bottom', 'beakcorner')
BeakAngle(BeakDist,'beak gap','lower side', 'upper side') # though this creates new variable it modifies BeakDist

# centered coordinates on beak corner
BCmedian, BCbased = beakCentered(BeakDf, 'beakcorner')

# kinematics, velocity, of the change in beak gap
GapKine = GapKinematics(BeakDist, 'beak gap')

# normalised
BeakgapNormal, BGcor, BGbaseline = nrL.CorrectBaseline(BeakDist,'beak gap', quantile = 0.005, mode='bottomup')

# smoothing
GapKine_sma = smL.SimpleMovingAvg_DF(GapKine, 4)


##############################################################################
### PLOTS ####################################################################


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

pBeakGap = plotOrg2Norm(BeakDist, BeakgapNormal, 'beak gap distance')

def plotOverTime(df, col,xname, yname):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(len(df)), df[col])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.title(yname+' over time')
    
    return fig

pBeakAngle = plotOverTime(BeakDist, 'beakcorner angle','frames', 'beak angle in rad')

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

pBeakSpatial = plot3InSpace(BCbased,((('beaktip_top','x'),('beaktip_top','y'),'upper beak'),
                                     (('beaktip_bottom','x'),('beaktip_bottom','y'),'lower beak'),
                                     (('beakcorner','x'),('beakcorner','y'),'beakcorner')), 'pxl','beak bp´s')




pBeakVelo_sma = plotOverTime(GapKine_sma, 'velocity pxl/s', 'frames', 'velocity pxl/s')
"""