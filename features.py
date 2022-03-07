# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:49:58 2021

@author: LeoBoeger
"""

# feature script
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
import time
from scipy import signal, stats

# own
import eyefeatures as eye
import beakfeatures as beak
from pylibLeo import ImgXtractForms as lxL
from pylibLeo import rotation as cpL
from pylibLeo import Smoothing as smL
from pylibLeo import Normalising as nrL
from pylibLeo import peakDetection as pkL
import plotfeatures as plot

##############################################################################
### SETTINGS #################################################################
### here you set the directory where the videos and the DLC coordinates (csv) are
### also you can set other relevant parameters like fps
### txtf is the file which defines for frames for each videos to extract for the eyelid rotation

dirIn = ''
dirbase_out = ''
csvFile = ''
video = ''
ID = '_'.join(video.split('_')[:2])
txtf = 'openeye.txt'
fps = 30

csv_dir_in = os.path.join(dirIn, csvFile)
csv_in = pd.read_csv(csv_dir_in, header=[1,2])
outname = "_".join(video.split('_')[:5]).split('.')[0]

dir_out = os.path.join(dirbase_out, ID)

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

bp_names = list(csv_in.columns)

### here the different body parts are splitted, simply for organising the data
Leye_col = []
for col in bp_names:
    for row in col:
        if "L_eye" in row or "L_pupil" in row:
            Leye_col.append(col)
            break
LEyeDf = csv_in[Leye_col]

Reye_col = []
for col in bp_names:
    for row in col:
        if "R_eye" in row or "R_pupil" in row:
            Reye_col.append(col)
            break
REyeDf = csv_in[Reye_col]

beak_col = []
for col in bp_names:
    for row in col:
        if "beak" in row:
            beak_col.append(col)
            break
BeakDf = csv_in[beak_col]

head_colL = []
head_colR = []
for col in bp_names:
    for row in col:
        if "L_head" in row:
            head_colL.append(col)
        if "R_head" in row:
            head_colR.append(col)
HeadDfL = csv_in[head_colL]
HeadDfR = csv_in[head_colR]

LlidXY = LEyeDf[[('L_eyelid','x'),('L_eyelid','y')]]
RlidXY = REyeDf[[('R_eyelid','x'),('R_eyelid','y')]]

##############################################################################
##############################################################################
### CALL F EYE ###############################################################
### the actual computaions are done in different files, for more detailed comments look there
### these files are: eyefeatures, beakfeatures
### additional helper functions are in ImageXtractForms, rotation, Smoothing, Normalising

# for later use, the angle of rotation is computed to rotate the coordinate system, so that the eyelid is parallel to x axis
# degVarLR is a tuple showing the variance of the left and right angle
degL, degR, degVarLR = lxL.drawXGetAngle(dirIn, video, ID, txtf)

# pupil diameter is computed from the 4 pupil tracking points
LpupilDiaBr, LpupilDiaBl, LpupilDia, xEL, yEL = eye.pupilDia(LEyeDf,  'L_pupil_tr', 'L_pupil_br', 'L_pupil_bl', 'L_pupil_tl')
RpupilDiaBr, RpupilDiaBl, RpupilDia, xER, yER = eye.pupilDia(REyeDf,  'R_pupil_tr', 'R_pupil_br', 'R_pupil_bl', 'R_pupil_tl')

# circle is fitted to pupil bodyparts, xEL / xER are arrays of the x coordinate of the 4 pupils of the left / right eye
# can be adapted: instead returning xEL, etc. in previous function, can be transformed from LEyeDf / REyeDf
# returns a dataframe with eye center coordinates and radius
Leye_odr = eye.eyeOdr(xEL, yEL)
Reye_odr = eye.eyeOdr(xER, yER)

# centrision of eye centers based on the geometric median
# geometric median is at 0 x and 0 y
LCmedian, LCbased = eye.eyeCentered(Leye_odr)
RCmedian, RCbased = eye.eyeCentered(Reye_odr)


# binning of coordinates,
# how often is the center in the bin?, z is frequency of binned
# binsize is 1 pxl
lx, ly, lz = eye.XYbinZ(LCbased, 'x from median', 'y from median')
rx, ry, rz = eye.XYbinZ(RCbased, 'x from median', 'y from median')

'''
# velocity of eye center
LCkinematics = eye.CenterKinematics(Leye_odr, fps)
RCkinematics = eye.CenterKinematics(Reye_odr, fps)
'''

# rotation of the previously centered eye centre locations, by previously acquired eyelid angle
# numpy array
LCrot = cpL.rotate_via_numpy(LCbased['x from median'], LCbased['y from median'],degL, deg=True)
RCrot = cpL.rotate_via_numpy(RCbased['x from median'], RCbased['y from median'],degR, deg=True)
# in order for both dfs to show the same directionality
RCrotf = np.concatenate((RCrot[:,0], -RCrot[:,1]), axis=1)
#pandas dataframe version of LCrot, RCrotf
LCrot_df = pd.DataFrame(LCrot, columns=['x rot','y rot'])
RCrot_df = pd.DataFrame(RCrotf, columns=['x rot','y rot'])

# smooth rotated, with window size 10
LCrot_sma = smL.SimpleMovingAvg_DF(LCrot_df, 10)
RCrot_sma = smL.SimpleMovingAvg_DF(RCrot_df, 10)

# kinematics of rotated smoothed data: euclidean distance and velocity
LCrotkine = eye.Kinematics(LCrot_sma,fps)
RCrotkine = eye.Kinematics(RCrot_sma,fps)
# smooth velocity, with window size 10
LCrotkine_sma = smL.SimpleMovingAvg_DF(LCrotkine, 10)
RCrotkine_sma = smL.SimpleMovingAvg_DF(RCrotkine, 10)

# Extraction of the x coordinate from the rotated smoothed data
LCrotX_sma = pd.DataFrame(LCrot_sma['x rot'])
RCrotX_sma =  pd.DataFrame(RCrot_sma['x rot'])
# Normalisation of the X rotated data
LCrotXNorm, LCrotXCor, LCrotXbaseline = nrL.CorrectBaseline(LCrotX_sma, 0)
RCrotXNorm, RCrotXCor, RCrotXbaseline = nrL.CorrectBaseline(RCrotX_sma, 0)


# baseline correction of pupil daiameters, first variable is corrected and normalised
LDiaNormalised, LDiaCor, LDiabaseline = nrL.CorrectBaseline(LpupilDia, 0)
RDiaNormalised, RDiaCor, RDiabaseline = nrL.CorrectBaseline(RpupilDia, 0)

# smooth normalised pupil daiameters
LDiaNorm_sma = smL.SimpleMovingAvg_DF(LDiaNormalised, 10)
RDiaNorm_sma = smL.SimpleMovingAvg_DF(RDiaNormalised, 10)

'''
LCbased_sma = smL.SimpleMovingAvg_DF(LCbased, 10, skip=['x from median', 'y from median'])
RCbased_sma = smL.SimpleMovingAvg_DF(RCbased, 10, skip=['x from median', 'y from median'])

LCkine_sma = smL.SimpleMovingAvg_DF(LCkinematics, 20, skip=['x center', 'y center'])
RCkine_sma = smL.SimpleMovingAvg_DF(RCkinematics, 20, skip=['x center', 'y center'])
'''

LLOpen_sm = eye.OpenEye(LEyeDf, degL)
RLOpen_sm = eye.OpenEye(REyeDf, -degR)

##############################################################################
### Peak detection of Eye ####################################################
### the script in peakDetection is called and the parameters are used to detect relevant peaks
### (org: DataFrame to detect peaks from; col: collumn to detect peaks from;
### *argv: some arguments which can be cheked: 'P1>Pnext' to detect peaks over second threshold only after P1
###  w_snr: window for SMA, moving std and moving snr
### distance: to next peak; w_prom: window to look for peak prominence,
### w_peaks: False, set to length of window, if only peaks with second peak within w_peaks should be considered;
### method: method of calculating the threshold, m = multiplicative factor of first threshold, n: of second; w_2nd: not used, same as distance
### MinP: minimal number of peaks in bursts; fps: frame per second

# inversion, because pupil constrictions should be detected as peaks
LDiaNorm_inv = (LDiaNorm_sma-100)*-1
# deteaction of pupil constriction peaks
LDpeaks, LDpeaksYs, LDpOnYs, LDpOffYs, LDsignal, LDthresh = pkL.PeakBoutDetect_Scipy(LDiaNorm_inv, 'diameter', w_snr=len(LDiaNorm_inv),distance = 10, w_prom = 100, method='new', m=3) #new2
pLDiaPeak = plot.plotPeakBouts(LDiaNorm_inv, LDpeaksYs, LDpOnYs, LDpOffYs, signal = LDsignal, title='T: median+amd*(median/amd)*2; left eye diameter', baseline=LDthresh)
RDiaNorm_inv = (RDiaNorm_sma-100)*-1
RDpeaks, RDpeaksYs, RDpOnYs, RDpOffYs, RDsignal, RDthresh = pkL.PeakBoutDetect_Scipy(RDiaNorm_inv, 'diameter', w_snr=len(RDiaNorm_inv), distance = 10, w_prom = 100, method='new', m=3)
pRDiaPeak = plot.plotPeakBouts(RDiaNorm_inv, RDpeaksYs, RDpOnYs, RDpOffYs, signal = RDsignal, title='T: median+amd*(median/amd)*2; right eye diameter', baseline=RDthresh)

# for peak detection nan values have to be filled with 0; first value of velocity (and euclidean distance) is naturally nan
LCrotvelo = pd.DataFrame(LCrotkine_sma['velocity pxl/s']).fillna(0)
LKpeaks, LKpeaksYs, LKpOnYs, LKpOffYs, LKsignal, LKthresh = pkL.PeakBoutDetect_Scipy(LCrotvelo, 'velocity pxl/s', addarg=['P1>Pnext'], w_snr=len(LCrotvelo), distance = 10, w_prom = 30, w_peaks=4, method='other', m=6, n=3) #new
pLKinePeak = plot.plotPeakBouts(LCrotvelo, LKpeaksYs, LKpOnYs, LKpOffYs, signal = LKsignal, title='T1: median+amd*(median/amd)*8 T2:*6; left eye velocity', baseline=LKthresh)
RCrotvelo = pd.DataFrame(RCrotkine_sma['velocity pxl/s']).fillna(0)
RKpeaks, RKpeaksYs, RKpOnYs, RKpOffYs, RKsignal, RKthresh = pkL.PeakBoutDetect_Scipy(RCrotvelo, 'velocity pxl/s', addarg=['P1>Pnext'], w_snr=len(RCrotvelo), distance = 10, w_prom = 30, w_peaks=4, method='other', m=6, n=3)
pRKinePeak = plot.plotPeakBouts(RCrotvelo, RKpeaksYs, RKpOnYs, RKpOffYs, signal = RKsignal, title='T1: median+amd*(median/amd)*8 T2:*6; right eye velocity', baseline=RKthresh)

"""
# peakas are not longer detected, because the baseline is very variable / oscilates
LLpeaks, LLpeaksYs, LLpOnYs, LLpOffYs, LLsignal, LLthresh = pkL.PeakBoutDetect_Scipy(LCrotXNorm, 'x rot', len(LCrotXNorm), 10, 30, method='poly', m=2) #new
pLKinePeak = plot.plotPeakBouts(LCrotXNorm, LLpeaksYs, LLpOnYs, LLpOffYs, signal = LLsignal, title='T1: median+amd*(median/amd)*8 T2:*6; left eye x location', baseline=LLthresh)
RLpeaks, RLpeaksYs, RLpOnYs, RLpOffYs, RLsignal, RLthresh = pkL.PeakBoutDetect_Scipy(RCrotXNorm, 'x rot', len(RCrotX_sma), 10, 30, method='sma+std*6', m=4) #new
pRLinePeak = plot.plotPeakBouts(RCrotX_sma, RLpeaksYs, RLpOnYs, RLpOffYs, signal = RLsignal, title='T1: median+amd*(median/amd)*8 T2:*6; left eye y location', baseline=RLthresh)
"""

##############################################################################
##############################################################################
### CALL F BEAK ##############################################################
# beak distance of all 3 bp´s + angle oover beak corner
BeakDist = beak.EuclDist(BeakDf,  'beaktip_top', 'beaktip_bottom', 'beakcorner')
# calculation of the angle between all 3 beak tracking points
# though this creates no new variable it modifies BeakDist
beak.BeakAngle(BeakDist,'beak gap','lower side', 'upper side')


# normalisation of beak gap (distance between upper and lower beak)
# mode='bottomup': baseline is close to 0
BeakgapNormal, BGcor, BGbaseline = nrL.CorrectBaseline(BeakDist,'beak gap', quantile = 0.005, mode='bottomup')

# kinematics (euclidean distance and velocity) of the beak gap
GapKine = beak.GapKinematics(BeakgapNormal, 'beak gap', fps)
# smoothed normalised data; not used
BGnormal_sma = smL.SimpleMovingAvg_DF(BeakgapNormal, 10)

##############################################################################
### Peak detection of Beak ###################################################
### see above for function details

# same as above, no nan value allowed for peak detection
BGvelo = pd.DataFrame(GapKine['velocity pxl/s']).fillna(0)
BGpeaks, BGpeaksYs, BGpOnYs, BGpOffYs, BGsignal, BGthresh = pkL.PeakBoutDetect_Scipy(BGvelo, 'velocity pxl/s', w_snr =len(BGvelo), distance = 0, w_prom = 30, w_peaks=1, method='other', m=15, n=5, MinP=4)
pBeakGap = plot.plotPeakBouts(BGvelo, BGpeaksYs, BGpOnYs, BGpOffYs, signal = BGsignal, title='T: (median+amd)*2; beak gap distance', baseline=BGthresh)


##############################################################################
##############################################################################
### CALL F HEAD ##############################################################
# head body parts are not used currently
"""
HLmedian, HLbased = beak.beakCentered(HeadDfL, 'L_head')
HRmedian, HRbased = beak.beakCentered(HeadDfR, 'R_head')

HLm, HLc, HLp1 = rgL.LinReg(HLbased)
HLdeg = math.atan2(HLp1[1],HLp1[0])*180/np.pi
HLrot = cpL.rotate_via_numpy(HLbased.iloc[:,0], HLbased.iloc[:,1],HLdeg, deg=True)
HLrotfh = np.concatenate((-HLrot[:,0], HLrot[:,1]), axis=1)

HRm, HRc, HRp1 = rgL.LinReg(HRbased)
HRdeg = math.atan2(HRp1[1],HRp1[0])*180/np.pi
HRrot = cpL.rotate_via_numpy(HRbased.iloc[:,0], HRbased.iloc[:,1],HRdeg, deg=True)
"""

##############################################################################
##############################################################################
### SAVING ###################################################################
# important left and right eye features
L_imp = [LDsignal,
         LKsignal]
prefinal_left = pd.concat(L_imp, axis=1).add_prefix('left ')
leftLoc = LCrotXNorm.where(prefinal_left.count(axis=1)>=1).add_prefix('left ')

R_imp = [RDsignal,
         RKsignal]
prefinal_right= pd.concat(R_imp, axis=1).add_prefix('right ')
rightLoc = LCrotXNorm.where(prefinal_right.count(axis=1)>=1).add_prefix('right ')

# addition of beak
final = pd.concat([prefinal_left, leftLoc, prefinal_right, rightLoc, BGsignal.add_prefix('beak ')], axis=1)#, BGsignal
final['REM all'] = np.where(final.count(axis=1)>=1, 1, 0)
final.to_csv(os.path.join(dir_out,outname+'.csv'))


##############################################################################
##############################################################################
### CALL Plots ###############################################################
### not many plots are created at the moment; used mainly for data exploration

# corellation of all features
# not optimal because data is time series; correlation based partly on temporary dynamics
to_correlate = [LDiaNorm_sma, LCrotvelo, LCrotXNorm, RDiaNorm_sma, RCrotvelo, RCrotXNorm, BGvelo]
nm_correlate = ['LDia', 'LKine', 'leftLoc', 'RDia', 'RKine', 'rightLoc', 'BGvelo']
corr_folder = 'C:/Users/labgrprattenborg/Desktop/example plots/corr'
randp = []
for x in to_correlate:
    for y in to_correlate:
        r, p = stats.pearsonr(x.values.flatten(),y.values.flatten())
        r,p = round(r,2), round(p,2)
        randp.append((r,p))
plot.plotCorrMultiple(to_correlate, nm_correlate)

"""
### Eyes
pLDiameter = plot.plotOrg2Norm(LpupilDia, LDiaNormalised,  'left pupil diameter')
pRDiameter = plot.plotOrg2Norm(RpupilDia, RDiaNormalised, 'right pupil diameter')
'''
pL3Dcenter = plot.plot3DCenter(lx, ly, lz, 'left')
pR3Dcenter = plot.plot3DCenter(rx, ry, rz, 'right')
'''
#pRVeloRadar = plotVeloRadar(RCbased, RCkinematics, 'right')
#pLVeloRadar = plotVeloRadar(LCbased, LCkinematics, 'left')
#pRDistRadar = plotDistRadar(RCbased, RCkinematics, 'right')
#pLDistRadar = plotDistRadar(LCbased, LCkinematics, 'left')

pLCbasedRot = plot.plotCbasedRot([LCrot[:,0]], [LCrot[:,1]], 'left eye', 'eyelid', 'back -> front', 'bottom -> top')
pRCbasedRot = plot.plotCbasedRot([RCrotf[:,0]], [RCrotf[:,1]], 'right eye', 'eyelid', 'back -> front', 'bottom -> top')

'''
pLKinematics = plot.plotKinematics(LCkine_sma,LCbased_sma, LCrot[:,0], 'left', 10, fps)
pRKinematics = plot.plotKinematics(RCkine_sma,RCbased_sma, RCrot[:,0], 'right', 10, fps)
'''

pCorrRotDir = plot.plotCorr([x for x in LCrot[:,0].flat], [x for x in RCrot[:,0].flat], 'x rotated movement, both eyes', 'left rotated x', 'right rotated x',  0, -2)
"""
"""
pLCorrODRDia = plot.plotCorr(LpupilDia, Leye_odr['radius'], 'left eye', 'weighted diameter in pxl', 'ODR based radius in pxl',  0, -2)
pRCorrODRDia = plot.plotCorr(RpupilDia, Reye_odr['radius'], 'right eye', 'weighted diameter in pxl', 'ODR based radius in pxl',  0, -2)

### Beak

pBeakGap = plot.plotScatNLine(BeakDist, BeakgapNormal, 'beak gap distance')
pBeakAngle = plot.plotOverTime(BeakDist, 'beakcorner angle','frames', 'beak angle in rad')
pBeakSpatial = plot.plot3InSpace(BCbased,((('beaktip_top','x'),('beaktip_top','y'),'upper beak'),
                                     (('beaktip_bottom','x'),('beaktip_bottom','y'),'lower beak'),
                                     (('beakcorner','x'),('beakcorner','y'),'beakcorner')), 'pxl','beak bp´s')
pBeakVelo_sma = plot.plotOverTime(GapKine, 'velocity pxl/s', 'frames', 'velocity pxl/s')

#pHLbasedRot = plot.plotCbasedRot([HLrotfh[:,0]], [HLrotfh[:,1]], 'left head', 'linear regression f(x) = mx+c', 'medial -> lateral','top -> bottom')
#pHRbasedRot = plot.plotCbasedRot([HRrot[:,0]], [HRrot[:,1]], 'right head', 'linear regression f(x) = mx+c', 'medial -> lateral','top -> bottom')


figures = [pLDiameter,pRDiameter,
           #pL3Dcenter,pR3Dcenter,
           pLCbasedRot,pRCbasedRot,
           #pLKinematics,pRKinematics,
           pLCorrODRDia,pRCorrODRDia,
           pBeakGap, pBeakAngle, pBeakSpatial, pBeakVelo_sma]


plot.write_pdf(os.path.join(dir_out,outname+'.tiff'), figures, outname)


##############################################################################
### Movie ####################################################################

#odr_data = Leye_odr.astype(int)
#vidm.CreateVideo(dirIn, video, odr_data)
"""
