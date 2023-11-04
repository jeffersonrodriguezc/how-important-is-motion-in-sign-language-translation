import numpy as np
import os
import shutil
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
from sklearn.metrics import pairwise_distances_argmin_min
from collections import defaultdict
from keras import backend as K

def normalizeSigns(path_videos, mean, path2save, type_='flow'):
    for f in os.listdir(path_videos):
        print("Folder: ", f)
        for l in os.listdir(path_videos+f):
            print("Video: ", l)
            n_frames_list = list()
            n_frames = len(sorted(os.listdir(path_videos+f+'/'+str(l)+'/'+type_+'/')))
            for i in sorted(os.listdir(path_videos+f+'/'+str(l)+'/'+type_+'/')):
                n_frames_list.append(io.imread(path_videos+f+'/'+str(l)+'/'+type_+'/'+str(i)))
            if n_frames > mean:
                n_frames_list = subsampling(n_frames_list, mean)
            elif n_frames < mean:
                n_frames_list = oversampling(n_frames_list, mean)
            VideoMatrix = np.array(n_frames_list)
            print("Final dim : ", VideoMatrix.shape)
            np.save(os.path.join(os.getcwd()+path2save, f+'/'+l),VideoMatrix)
    
def subsampling(sign_list, mean=128):
    print("Subsampling ...")
    flag = True
    while flag:
        len_sign = len(sign_list)
        dif =  len_sign - int(mean)
        if dif != 0:
            step = int(np.ceil(len_sign/dif))
            sub = [i for ind, i in enumerate(sign_list) if ind%step!=0]
            sign_list = sub
            if len(sign_list) == mean:
                flag = False
        else:
            flag = False
  
    return sign_list

def oversampling(sign_list, mean=78):
    print("Oversampling ...")
    copy_list = np.copy(sign_list)
    len_sign = len(sign_list)
    dif =  int(mean) - len_sign
    step = int(np.floor(len_sign/dif))
    count = 1
    for i in range(1,len_sign-1,step):
        if count <= dif:
            oimg = np.around((copy_list[i] + copy_list[i+1])/2, 4)
            sign_list.insert(i-1, oimg)
            count += 1
    if len(sign_list) != mean:
        oimg = np.around((copy_list[-1] + copy_list[-2])/2,4)
        sign_list.insert(-1, oimg)
    return sign_list

