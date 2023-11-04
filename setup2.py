#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Jefferson Rodríguez"
__copyright__ = ""
__credits__ = ["Jefferson Rodríguez", "Fabio Martinez"]
__license__ = "GNU GPL"
__version__ = "1.0"
__maintainer__ = "Jefferson Rodríguez"

import argparse
import multiprocessing as mp
import matplotlib.image as pimage
import numpy as np
import os
import shutil
from collections import defaultdict
import cv2
from time import time
import pickle
import matplotlib.pyplot as plt
from utils.utils import *
from utils.augmentation import *
import pandas as pd

def worker_recursive(arg):
    path,args = arg
    inf = path.split('/')[-4:-1]
    print(inf) 
    frames = [pimage.imread(path+i) for i in sorted(os.listdir(path))]
    # Data Augmentation
    clips = frame_sampling(frames, args.nTempo)
    if args.flipV==True and args.frameSampling==True:
        for j, c in enumerate(clips):
            np.save(args.path2save+inf[0]+'/'+inf[-1]+'/'+inf[1]+'_'+str(j),c)
            c_flip = flip_vertical(c)
            np.save(args.path2save+inf[0]+'/'+inf[-1]+'/'+inf[1]+'_'+str(j)+'_flip',c_flip)
    elif args.flipV==True and args.frameSampling==False:
        np.save(args.path2save+inf[0]+'/'+inf[-1]+'/'+inf[1],clips[0])
        c_flip = flip_vertical(clips[0])
        np.save(args.path2save+inf[0]+'/'+inf[-1]+'/'+inf[1]+'_flip',c_flip)
    elif args.flipV==False and args.frameSampling==True:
        for j, c in enumerate(clips):
            np.save(args.path2save+inf[0]+'/'+inf[-1]+'/'+inf[1]+'_'+str(j),c)
    else:
        np.save(args.path2save+inf[0]+'/'+inf[-1]+'/'+inf[1],clips[0])
    
    return True

def main():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--path', type=str,  
                        default= '../Datasets/Pre-procesados/SLR/RWTH-Phoenix/phoenixT/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/frame_features/227/',
                        help='path of videos (train, test, valid)')
    parser.add_argument('--type', type=str,  
                        default= 'flow',
                        help='kind of images')
    parser.add_argument('--workers', type=int,  
                        default=8,
                        help='number of workers')
    parser.add_argument('--path2save', type=str,  
                        default= os.getcwd()+'/results/dataTrain_phoenix_210x260/',
                        help='path to save the images')
    parser.add_argument('--pathAnnotations', type=str,  
                        default= '../Datasets/Pre-procesados/SLR/RWTH-Phoenix/phoenixT/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/annotations.csv',
                        help='path of annotations file')
    parser.add_argument('--flipV', type=bool,  
                        default=False,
                        help='flip operation (vertical)')
    parser.add_argument('--frameSampling', type=bool,  
                        default=False,
                        help='frame sampling for data augmentation')
    parser.add_argument('--nTempo', type=int,  
                        default=60,
                        help='number of temporal frames ')
    
    args = parser.parse_args()
    
    def run(args, worker):
        tic = time()
        for f in ['train', 'test', 'dev']:#os.listdir(args.path):
            print('folder: ', f)
            pt = args.path+f+'/'
            videosargs = [(pt+v+'/'+args.type+'/',args) for v in os.listdir(pt)]
            pool = mp.Pool(processes=args.workers)
            result = [pool.map(worker_recursive,videosargs)]
            pool.close()    
        tac = time()
        print(tac - tic , ' sg')
        return True
    
    run(args, worker_recursive)

    
if __name__ == '__main__':
    main()
    
