# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:27:43 2019

@author: hoss3301
"""

import WORLD
from WORLD.world import main
import numpy as np
import h5py, os
import scipy.io


h5_folder = './data/h5_sounds/'
if not os.path.isdir(h5_folder):
    os.mkdir(h5_folder)   


dataPath = './data/original/input.mat'
mat1 = scipy.io.loadmat(dataPath)
fs=97656.25
for k in ['train','test']:
    data = mat1['input_' + k]
    print(k, data.shape)
    
    
    for i in range(data.shape[1]):
        data[:,i]=data[:,i]/(np.max(data[:,i]))
    
    x = np.concatenate([sound for sound in np.transpose(data)])

#wav_path = Path('C:/Users/hoss3301/work/WORLD/test/test-mwm.wav')
#fs, x_int16 = wavread(wav_path)


    vocoder = main.World()

    dat= vocoder.encode(fs, x, f0_method='dio',target_fs=32552,frame_period=20,allowed_range=0.2, is_requiem=True)
#dat = vocoder.encode(fs, x_int16, f0_method='dio', is_requiem=True)
    sp=dat['spectrogram']
    sp=sp.swapaxes(1,0)

    vuv=dat['vuv']
    vuv = np.expand_dims(vuv, axis=1)

    ap=dat['aperiodicity']
    ap=ap[0,:]
    ap=np.expand_dims(ap,axis=1)

    f0=dat['f0']
    f0 = np.expand_dims(f0, axis=1)
    
    print(f0.shape, vuv.shape, ap.shape, sp.shape)
    
    hf = h5py.File(h5_folder + 'clean_guinea_sounds_' + k + '.h5', 'w')
    hf.create_dataset('f0', data=f0)
    hf.create_dataset('vuv', data=vuv)
    hf.create_dataset('aperiodicity', data=ap)
    hf.create_dataset('spectrogram', data=sp)
    hf.close()
    
    
