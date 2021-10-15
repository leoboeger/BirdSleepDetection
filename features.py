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

# own
import eyefeatures as eye
import beakfeatures as beak
from pylibLeo import ImgXtractForms as lxL
from pylibLeo import rotation as cpL
from pylibLeo import Smoothing as smL
from pylibLeo import Normalising as nrL
from pylibLeo import Regression as rgL
from pylibLeo import peakDetection as pkL
import plotfeatures as plot

##############################################################################
### SETTINGS #################################################################

dirIn = 'C:/Users/labgrprattenborg/DeepLabCut/DLCprojects/test_videos/fMRIsleepBudapestResNet101Contrast-Leo-2021-04-20/iteration 0_1'
csvFile = 'Blue69_S2_Video_2021-01-12_09-30-00-000DLC_resnet_101_fMRIsleepBudapestResNet101ContrastApr14shuffle1_400000.csv'
video = "Blue69_S2_Video_2021-01-12_09-30-00-000.avi"
ID = '_'.join(video.split('_')[:2])
txtf = 'openeye.txt'
fps = 30

csv_dir_in = os.path.join(dirIn, csvFile)
csv_in = pd.read_csv(csv_dir_in, header=[1,2])
outname = "_".join(video.split('_')[:5]).split('.')[0]
dirbase_out = 'C:/Users/labgrprattenborg/DeepLabCut/DLCprojects/analysis/'
dir_out = os.path.join(dirbase_out, ID)

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

bp_names = list(csv_in.columns)

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


##############################################################################
### CALL F EYE ###############################################################
# pupil diameter

LpupilDiaBr, LpupilDiaBl, LpupilDia, xEL, yEL = eye.pupilDia(LEyeDf,  'L_pupil_tr', 'L_pupil_br', 'L_pupil_bl', 'L_pupil_tl')
RpupilDiaBr, RpupilDiaBl, RpupilDia, xER, yER = eye.pupilDia(REyeDf,  'R_pupil_tr', 'R_pupil_br', 'R_pupil_bl', 'R_pupil_tl')

# circle fit to pupil, with eye center and radius
Leye_odr = eye.eyeOdr(xEL, yEL)
Reye_odr = eye.eyeOdr(xER, yER)

# centrision of eye center
LCmedian, LCbased = eye.eyeCentered(Leye_odr)
RCmedian, RCbased = eye.eyeCentered(Reye_odr)



# 3D, binning of coordinates, z is frequency of binned
lx, ly, lz = eye.XYbinZ(LCbased, 'x from median', 'y from median')
rx, ry, rz = eye.XYbinZ(RCbased, 'x from median', 'y from median')

# velocity of eye center
LCkinematics = eye.CenterKinematics(Leye_odr, fps)
RCkinematics = eye.CenterKinematics(Reye_odr, fps)

# get rotation angle for left and rigth eye
degL, degR, degVarLR = lxL.drawXGetAngle(dirIn, video, ID, txtf)
LCrot = cpL.rotate_via_numpy(LCbased['x from median'], LCbased['y from median'],degL, deg=True)
RCrot = cpL.rotate_via_numpy(RCbased['x from median'], RCbased['y from median'],degR, deg=True)
RCrotf = np.concatenate((RCrot[:,0], -RCrot[:,1]), axis=1)
# smooth rotated
LCrot_sma = smL.SimpleMovingAvg_DF(pd.DataFrame(LCrot, columns=['x rot','y rot']), 10)
RCrot_sma = smL.SimpleMovingAvg_DF(pd.DataFrame(RCrotf, columns=['x rot','y rot']), 10)
# velocity of rotated
LCrotkine = eye.Kinematics(LCrot_sma,fps)
RCrotkine = eye.Kinematics(RCrot_sma,fps)


# baseline correction, first variable is corrected and normalised
LDiaNormalised, LDiaCor, LDiabaseline = nrL.CorrectBaseline(LpupilDia, col=0)
RDiaNormalised, RDiaCor, RDiabaseline = nrL.CorrectBaseline(RpupilDia, col=0)

# smoothing with SMA, arguments: df, window size, skippable columns(optinal)
LDiaNorm_sma = smL.SimpleMovingAvg_DF(LDiaNormalised, 10)
RDiaNorm_sma = smL.SimpleMovingAvg_DF(RDiaNormalised, 10)

LCbased_sma = smL.SimpleMovingAvg_DF(LCbased, 10, skip=['x from median', 'y from median'])
RCbased_sma = smL.SimpleMovingAvg_DF(RCbased, 10, skip=['x from median', 'y from median'])

LCkine_sma = smL.SimpleMovingAvg_DF(LCkinematics, 20, skip=['x center', 'y center'])
RCkine_sma = smL.SimpleMovingAvg_DF(RCkinematics, 20, skip=['x center', 'y center'])



# peak detection
LDiaNorm_inv = (LDiaNorm_sma-100)*-1
#from pylibLeo import peakDetection as pkL
LDpeaksYs, LDpOnYs, LDpOffYs, LDsignal, LDthresh = pkL.PeakBoutDetect_Scipy(LDiaNorm_inv, 'diameter', 40000, 20, method='other')
pLDiaPeak = plot.plotPeakBouts(LDiaNorm_inv, LDpeaksYs, LDpOnYs, LDpOffYs, signal = LDsignal, title='left eye diameter', baseline=LDthresh)
RDiaNorm_inv = (RDiaNorm_sma-100)*-1
RDpeaksYs, RDpOnYs, RDpOffYs, RDsignal, RDthresh = pkL.PeakBoutDetect_Scipy(RDiaNorm_inv, 'diameter', 40000, 20, method='other')
pRDiaPeak = plot.plotPeakBouts(RDiaNorm_inv, RDpeaksYs, RDpOnYs, RDpOffYs, signal = RDsignal, title='left eye diameter', baseline=RDthresh)

LCrotvelo = pd.DataFrame(LCrotkine['velocity pxl/s'])
LKpeaksYs, LKpOnYs, LKpOffYs, LKsignal, LKthresh = pkL.PeakBoutDetect_Scipy(LCrotvelo, 'velocity pxl/s', 40000, 10, 3, method='sma+std*6')
pLKinePeak = plot.plotPeakBouts(LCrotvelo, LKpeaksYs, LKpOnYs, LKpOffYs, signal = LKsignal, title='left eye velocity', baseline=LKthresh)
RCrotvelo = pd.DataFrame(RCrotkine['velocity pxl/s'])
RKpeaksYs, RKpOnYs, RKpOffYs, RKsignal, RKthresh = pkL.PeakBoutDetect_Scipy(RCrotvelo, 'velocity pxl/s', 40000, 10, 3, method='sma+std*6')
pRKinePeak = plot.plotPeakBouts(RCrotvelo, RKpeaksYs, RKpOnYs, RKpOffYs, signal = RKsignal, title='left eye velocity', baseline=RKthresh)



### CALL F BEAK ##############################################################
# beak distance of all 3 bp´s + angle oover beak corner
BeakDist = beak.EuclDist(BeakDf,  'beaktip_top', 'beaktip_bottom', 'beakcorner')
beak.BeakAngle(BeakDist,'beak gap','lower side', 'upper side') # though this creates new variable it modifies BeakDist

# centered coordinates on beak corner
BCmedian, BCbased = beak.beakCentered(BeakDf, 'beakcorner')

# kinematics, velocity, of the change in beak gap
GapKine = beak.GapKinematics(BeakDist, 'beak gap', fps)

# normalised
BeakgapNormal, BGcor, BGbaseline = nrL.CorrectBaseline(BeakDist,'beak gap', quantile = 0.005, mode='bottomup')


# peak detection, signal detection ###########################################
BGpeaksYs, BGpOnYs, BGpOffYs, BGsignal, BGthresh = pkL.PeakBoutDetect_Scipy(BeakgapNormal, 'beak gap', 40000, 10)
pBeakGap = plot.plotPeakBouts(BeakgapNormal, BGpeaksYs, BGpOnYs, BGpOffYs, signal = BGsignal, title='beak gap distance', baseline=BGthresh)



# smoothing
GapKine_sma = smL.SimpleMovingAvg_DF(GapKine, 10)
BGnormal_sma = smL.SimpleMovingAvg_DF(BeakgapNormal, 10)


### CALL F HEAD ##############################################################
# beak distance of all 3 bp´s + angle oover beak corner
 
HLmedian, HLbased = beak.beakCentered(HeadDfL, 'L_head')
HRmedian, HRbased = beak.beakCentered(HeadDfR, 'R_head')

HLm, HLc, HLp1 = rgL.LinReg(HLbased)
HLdeg = math.atan2(HLp1[1],HLp1[0])*180/np.pi
HLrot = cpL.rotate_via_numpy(HLbased.iloc[:,0], HLbased.iloc[:,1],HLdeg, deg=True)
HLrotfh = np.concatenate((-HLrot[:,0], HLrot[:,1]), axis=1)

HRm, HRc, HRp1 = rgL.LinReg(HRbased)
HRdeg = math.atan2(HRp1[1],HRp1[0])*180/np.pi
HRrot = cpL.rotate_via_numpy(HRbased.iloc[:,0], HRbased.iloc[:,1],HRdeg, deg=True)



##############################################################################
### SAVING ###################################################################
L_imp = [LDsignal,
         LCrot_sma,
         LKsignal['velocity pxl/s'],
         LCrotvelo]
prefinal_left = pd.concat(L_imp, axis=1).add_prefix('left_')

R_imp = [RDsignal, 
         RCrot_sma,
         RKsignal['velocity pxl/s'],
         RCrotvelo]
prefinal_right= pd.concat(R_imp, axis=1).add_prefix('right_')

beak_imp = [BeakgapNormal['beak gap'],
            BGsignal['beak gap'],
            GapKine['velocity pxl/s'],
            GapKine_sma['velocity pxl/s']]
prefinal_beak = pd.concat(beak_imp, axis=1)

head_imp = [pd.DataFrame(HLrotfh[:,1], columns=['rot y left head']),
            pd.DataFrame(HRrot[:,1], columns=['rot y right head'])]
prefinal_head = pd.concat(head_imp, axis=1)

final = pd.concat([prefinal_left,prefinal_right, prefinal_beak, prefinal_head], axis=1)
final.to_csv(os.path.join(dir_out,outname+'.csv'))


##############################################################################
### CALL Plots ###############################################################

### Eyes
pLDiameter = plot.plotOrg2Norm(LpupilDia, LDiaNormalised,  'left pupil diameter') 
pRDiameter = plot.plotOrg2Norm(RpupilDia, RDiaNormalised, 'right pupil diameter')  
pL3Dcenter = plot.plot3DCenter(lx, ly, lz, 'left')
pR3Dcenter = plot.plot3DCenter(rx, ry, rz, 'right')

#pRVeloRadar = plotVeloRadar(RCbased, RCkinematics, 'right')
#pLVeloRadar = plotVeloRadar(LCbased, LCkinematics, 'left')
#pRDistRadar = plotDistRadar(RCbased, RCkinematics, 'right')
#pLDistRadar = plotDistRadar(LCbased, LCkinematics, 'left')

pLCbasedRot = plot.plotCbasedRot([LCrot[:,0]], [LCrot[:,1]], 'left eye', 'eyelid', 'back -> front', 'bottom -> top')
pRCbasedRot = plot.plotCbasedRot([RCrotf[:,0]], [RCrotf[:,1]], 'right eye', 'eyelid', 'back -> front', 'bottom -> top')

pLKinematics = plot.plotKinematics(LCkine_sma,LCbased_sma, LCrot[:,0], 'left', 10, fps)
pRKinematics = plot.plotKinematics(RCkine_sma,RCbased_sma, RCrot[:,0], 'right', 10, fps)

pCorrRotDir = plot.plotCorr([x for x in LCrot[:,0].flat], [x for x in RCrot[:,0].flat], 'x rotated movement, both eyes', 'left rotated x', 'right rotated x',  0, -2)

pLCorrODRDia = plot.plotCorr(LpupilDia, Leye_odr['radius'], 'left eye', 'weighted diameter in pxl', 'ODR based radius in pxl',  0, -2)
pRCorrODRDia = plot.plotCorr(RpupilDia, Reye_odr['radius'], 'right eye', 'weighted diameter in pxl', 'ODR based radius in pxl',  0, -2)

### Beak

pBeakGap = plot.plotScatNLine(BeakDist, BeakgapNormal, 'beak gap distance')
pBeakAngle = plot.plotOverTime(BeakDist, 'beakcorner angle','frames', 'beak angle in rad')
pBeakSpatial = plot.plot3InSpace(BCbased,((('beaktip_top','x'),('beaktip_top','y'),'upper beak'),
                                     (('beaktip_bottom','x'),('beaktip_bottom','y'),'lower beak'),
                                     (('beakcorner','x'),('beakcorner','y'),'beakcorner')), 'pxl','beak bp´s')
pBeakVelo_sma = plot.plotOverTime(GapKine_sma, 'velocity pxl/s', 'frames', 'velocity pxl/s')

pHLbasedRot = plot.plotCbasedRot([HLrotfh[:,0]], [HLrotfh[:,1]], 'left head', 'linear regression f(x) = mx+c', 'medial -> lateral','top -> bottom')
pHRbasedRot = plot.plotCbasedRot([HRrot[:,0]], [HRrot[:,1]], 'right head', 'linear regression f(x) = mx+c', 'medial -> lateral','top -> bottom')


figures = [pLDiameter,pRDiameter,
           pL3Dcenter,pR3Dcenter,
           pLCbasedRot,pRCbasedRot,
           pLKinematics,pRKinematics,
           pLCorrODRDia,pRCorrODRDia,
           pBeakGap, pBeakAngle, pBeakSpatial, pBeakVelo_sma]


plot.write_pdf(os.path.join(dir_out,outname+'.tiff'), figures, outname)


##############################################################################
### Movie ####################################################################

#odr_data = Leye_odr.astype(int)
#vidm.CreateVideo(dirIn, video, odr_data)
