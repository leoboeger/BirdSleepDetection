# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 08:47:49 2021

@author: LeoBoeger
"""
##############################################################################
### IMPORT ###################################################################
# standard
import os
import sys
sys.path.append(os.path.expanduser('~'))
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import math
import copy
from scipy import stats

# third party
from tqdm import trange

# own
from pylibLeo import ODR_CenterRadius as odr
#from pylibLeo import VideoMaker as vidm
from pylibLeo import GeoMedian as gmL



##############################################################################
### FUNCTIONS ################################################################

### Diameter #################################################################
### Calculation of the euclidean distance for diameter
def pupilDia(df, A, B, C, D):
    dia_a2c = np.sqrt((df[(A, 'x')]-df[(C, 'x')])**2+
                       (df[(A, 'y')]-df[(C, 'y')])**2)
    dia_b2d = np.sqrt((df[(B, 'x')]-df[(D, 'x')])**2+
                       (df[(B, 'y')]-df[(D, 'y')])**2)
    
    p_a2c = (df[(A, 'likelihood')]+df[(C, 'likelihood')])/2
    p_b2d = (df[(B, 'likelihood')]+df[(D, 'likelihood')])/2
    
    dia = pd.DataFrame()
    dia = (dia_a2c*p_a2c + dia_b2d*p_b2d)/(p_a2c + p_b2d)
    dia_df = dia.to_frame(name = 'diameter')
    
    x_list = df[[(A, 'x'),(B, 'x'),(C, 'x'),(D, 'x')]].to_numpy(dtype='int32')
    y_list = df[[(A, 'y'),(B, 'y'),(C, 'y'),(D, 'y')]].to_numpy(dtype='int32')
    
    return dia_a2c, dia_b2d, dia_df, x_list, y_list



### Center (and Diameter) ####################################################
### fitting circle to pupil coordinates 
def eyeOdr(x_arr, y_arr):
    eye_odr = pd.DataFrame(columns=['x center', 'y center','radius'])
    trng = trange(len(x_arr))
    for frame in trng:
        trng.set_description('ODR, fitting a circle to the pupil')
        x_c, y_c, r_c = odr.ODRcircle(x_arr[frame], y_arr[frame])
        eye_odr.loc[frame] = [x_c, y_c, r_c]
    
    return eye_odr


### Centralisation on Geometric Median
def eyeCentered(center_df):
    center_arr = center_df[["x center", "y center"]].values
    base = gmL.geometric_median(center_arr)
    
    ### Coordinates of all frames transforemd on base
    c_based = pd.DataFrame(columns=['x from median','y from median','angle','eucl dist'])
    c_based['x from median'] = center_arr[:,0]-base[0]
    c_based['y from median'] = center_arr[:,1]-base[1]
    for coor in trange(len(c_based)):
        c_based.iloc[coor,2] = math.atan2(c_based.iloc[coor,1], c_based.iloc[coor,0])
    c_based['eucl dist'] = np.sqrt(+c_based['x from median']**2+c_based['y from median']**2)

    return base, c_based


### binning Coordinates and returning frequency as z
def XYbinZ(df, xcol, ycol, keep0s=False):
    xstretch = range(math.floor(min(df[xcol])), math.ceil(max(df[xcol])+1), 1)
    ystretch = range(math.floor(min(df[ycol])), math.ceil(max(df[ycol])+1),1)
    ret = stats.binned_statistic_2d(df[xcol], df[ycol], None, 'count', bins=[xstretch, ystretch], expand_binnumbers=True)
    ret2 = ret.statistic
    xedge = ret.x_edge
    yedge = ret.y_edge
        
    xarr = np.repeat(xedge[:-1], len(yedge)-1)
    yarr = np.tile(yedge[:-1], len(xedge)-1)
    zarr = np.ravel(ret2)
    
    if not keep0s:
        ForgetNan = np.array(zarr==0)
        xarr = np.delete(xarr,ForgetNan)
        yarr = np.delete(yarr,ForgetNan)
        zarr = np.delete(zarr,ForgetNan)
        
        
    return  xarr, yarr, zarr



### Kinematics ###############################################################
### euclidean distance to previous frame and velocity
def CenterKinematics(center_df, fps):
    kine_df = center_df[['x center', 'y center']]
    help_df = pd.DataFrame()
    help_df[["x c prev","y c prev"]] = kine_df[['x center', 'y center']].shift(1)
    kine_df['eucl dist prev'] = np.sqrt((kine_df['x center']-help_df["x c prev"])**2 +
                      (kine_df['y center']-help_df["y c prev"])**2)
    kine_df['velocity pxl/s'] = kine_df['eucl dist prev']*(1000/fps)

    return kine_df

def Kinematics(df, fps):
    cols = df.columns
    xcol,ycol = [x for x in cols if 'x' in x], [x for x in cols if 'y' in x]
    if len(xcol) >= 2 or len(ycol) >= 2:
        print('Warning! collumn identification was ambigious\nFirst column was used')
    kine_df = df[[xcol[0], ycol[0]]]
    help_df = pd.DataFrame()
    help_df[["x prev","y prev"]] = kine_df[[xcol[0], ycol[0]]].shift(1)
    kine_df['eucl dist prev'] = np.sqrt((kine_df[xcol[0]]-help_df["x prev"])**2 +
                      (kine_df[ycol[0]]-help_df["y prev"])**2)
    kine_df['velocity pxl/s'] = kine_df['eucl dist prev']*(1000/fps)

    return kine_df


### Adjusting ################################################################
### sliding mean of # frames
def SimpleMovingAvg(df, window, skip=[]):
    sma_df = pd.DataFrame()
    try:
        dfcol = df.columns.to_list()
        for col in dfcol:
            if col not in skip:
                idx = df.columns.get_loc(col)
                sma_df[col] = df.iloc[:,idx].rolling(window=window, min_periods=1, center=True).median()
    
    except:
        truedf = df.to_frame()
        sma_df[0] = truedf.iloc[:,0].rolling(window=window, min_periods=1, center=True).median()
    
    return sma_df


### Baseline Correction with SMA, Normalisation to Percent
def CorrectBaseline(df, col, window = 10000, quantile = 0.995, mode='topdown'):
    if not isinstance(col, int):
        col_idx = df.columns.get_loc(col)
        col = col_idx
    # get baseline to adjust data to
    baseline = SimpleMovingAvg(df, window)
    
    # for normalisation
    # get the max data, without outlier, as done with high quantile
    cutat = df.iloc[:,col].quantile(quantile)
    # calculate the difference of cutat to the baseline at its index
    cutat_idx = (df.iloc[:,col].sort_values(ascending=False) <= df.iloc[:,col].quantile(quantile)).idxmax()
    cutatbase_dif = cutat - baseline.iloc[cutat_idx,col]
    # get the lowest datapoint
    if mode == "topdown":
        lower = min(df.iloc[:,col])
    elif mode == 'bottomup':
        lower = max(df.iloc[:,col])
    
    # correct df with substraction of the baseline
    # + cutat - cutatbase_dif ensures that data is distributed in a similar range
    corrected = (df.iloc[:,col].sub(baseline.iloc[:,col])+ cutat - cutatbase_dif).to_frame()
    if mode == "topdown":
        normalised = (corrected - lower)*100/(cutat - lower)
    if mode == "bottomup":
        normalised = (corrected - cutat)*100/(lower - cutat)
    
    return normalised, corrected, baseline


### Filtering ################################################################
### Thresholding
def Thresh_TimeConserved(df2filt, dfcol4bool, thresh, mode):
    df_filt = copy.deepcopy(df2filt)

    if mode == '<':
        df_filt.loc[dfcol4bool<thresh] = np.nan
    
    return df_filt


