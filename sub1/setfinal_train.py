import os
from pydub import AudioSegment, effects

import librosa
import numpy as np
import copy


os.system('rm -rf set_final')
os.system('mkdir set_final')
os.system('mkdir solicited_norm')

REF=open('CODA_TB_Solicited_Meta_Info.csv','r')
REF.readline()
ref={}
for lll in REF:
    lll=lll.strip()
    ttt=lll.split(',')
    if ttt[0] in ref:
        ref[ttt[0]]=ref[ttt[0]]+1
    else:
        ref[ttt[0]]=1
REF.close()

REF=open('CODA_TB_Solicited_Meta_Info.csv','r')
REF.readline()
for lll in REF:
    lll=lll.strip()
    ttt=lll.split(',')
    ### original data for two deltas
    y,sr=librosa.load('./solicited/'+ttt[1])
    feature=np.zeros((20,22))
    feature_tmp=librosa.feature.mfcc(y=y, sr=sr)
    feature_tmp=librosa.feature.delta(feature_tmp)
    feature[0:feature_tmp.shape[0],0:feature_tmp.shape[1]]=feature_tmp
    feature=feature.flatten()
    full_data=copy.copy(feature)

    feature=np.zeros((20,22))
    feature_tmp=librosa.feature.mfcc(y=y, sr=sr)
    feature_tmp=librosa.feature.delta(feature_tmp,order=2)
    feature[0:feature_tmp.shape[0],0:feature_tmp.shape[1]]=feature_tmp
    feature=feature.flatten()
    full_data=np.hstack((full_data,feature))


    ### normalize data
    rawsound = AudioSegment.from_file('./solicited/'+ttt[1], "wav")
    normalizedsound = effects.normalize(rawsound)
    normalizedsound.export("solicited_norm/"+ttt[1], format="wav")

    y, sr = librosa.load('solicited_norm/'+ttt[1])
    feature=np.zeros((20,22))
    feature_tmp=librosa.feature.mfcc(y=y, sr=sr)
    feature[0:feature_tmp.shape[0],0:feature_tmp.shape[1]]=feature_tmp
    feature=feature.flatten()
    full_data=np.hstack((full_data,feature))

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    feature=np.zeros((1025,22))
    feature_tmp, mag_abc=librosa.piptrack(y=y, sr=sr)
    feature[0:mag_abc.shape[0],0:mag_abc.shape[1]]=mag_abc
    feature=feature.flatten()
    full_data=np.hstack((full_data,feature))
    full_data=np.hstack((full_data,ref[ttt[0]]))

    np.save('set_final/'+ttt[1],full_data)
