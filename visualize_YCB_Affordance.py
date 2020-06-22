import os
import argparse
import glob
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np

import time
from collections import OrderedDict
import os

from manopth import rodrigues_layer
from manopth.manolayer import ManoLayer
from manopth.tensutils import th_posemap_axisang, make_list, th_pack, subtract_flat_id, th_with_zeros

from matplotlib import pyplot as plt

import tqdm

from utils import fast_load_obj, plot_scene_w_grasps
from scipy.stats import pearsonr
import scipy.io

def Visualize():
    # Initialize MANO layer
    MANO = ManoLayer(
            mano_root='data/mano/', side='right', use_pca=True, ncomps=45, flat_hand_mean=True)


    # set model to eval

    mano_faces = MANO.th_faces.cpu().data.numpy()

    # Analysis:
    results_angles = []
    results_percentage_visible_points = []

    ## YCB INFO:
    #imgnames = np.load('data/imgnames.npy')
    imgnames = glob.glob('data/YCB_Video_Dataset/data/????/*color.png')
    imgnames.sort()
    models = glob.glob('data/models/*')
    models.sort()
    # Annotated plane equations
    planes = np.load('data/plane_equations.npy')
    # Which objs from the YCB object set that are in YCB-Videos
    objects_in_YCB = np.load('data/objects_in_YCB.npy')

    # offsets in the ycb video dataset
    offset_ycbs = np.load('data/offsets.npy') #translation_between_YCBObjects_and_YCBVideos.npy')
    # Load object shape for the 21 objs that are in YCB-Video

    obj_verts = []
    obj_faces = []
    for index, i in enumerate(objects_in_YCB):
        obj = fast_load_obj(open(models[i]+'/google_16k/textured.obj', 'rb'))[0]
        obj_verts.append(obj['vertices'] - offset_ycbs[index])
        obj_faces.append(obj['faces'])

    # Render 3D scenes with input image
    random_indexs = np.arange(len(imgnames))
    np.random.shuffle(random_indexs)

    for ind in range(len(random_indexs)):
        i = random_indexs[ind]
        imgname = imgnames[i]
        plane = planes[i]

        meta = scipy.io.loadmat(str.split(imgname, 'color.png')[0] + 'meta.mat')

        object_ids = meta['cls_indexes'][:, 0] - 1
        numobjects = len(object_ids)

        scene_hands_verts = []
        scene_hands_faces = []
        scene_obj_verts = []
        scene_obj_faces = []

        hand_translations = np.load('data/YCB_Affordance_grasps/mano_translation_%d.npy'%(i), allow_pickle=True)
        hand_representation = np.load('data/YCB_Affordance_grasps/mano_representation_%d.npy'%(i), allow_pickle=True)
        for j in range(numobjects):
            rt = meta['poses'][:, :, j]

            p = np.matmul(rt[:3,0:3], obj_verts[object_ids[j]].T) + rt[:3,3].reshape(-1,1)
            p = p.T

            scene_obj_verts.append(p)
            scene_obj_faces.append(obj_faces[object_ids[j]])

            if len(hand_translations[j]) == 0: # Object is ungraspable
                continue

            # Only plot the first annotated grasp solution for objects:
            trans = torch.FloatTensor(hand_translations[j][0]).unsqueeze(0)
            rep = torch.FloatTensor(hand_representation[j][0]).unsqueeze(0)

            verts = MANO(rep, th_trans=trans)[0][0].cpu().data.numpy()/1000

            scene_hands_verts.append(verts)
            scene_hands_faces.append(mano_faces)

        print("Press a key to visualize another scene:")
        plot_scene_w_grasps(scene_obj_verts, scene_obj_faces, scene_hands_verts, scene_hands_faces, plane)


if __name__ == '__main__':
    Visualize()
