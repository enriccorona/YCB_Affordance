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
import tqdm

from manopth import rodrigues_layer
from manopth.tensutils import th_posemap_axisang, make_list, th_pack, subtract_flat_id, th_with_zeros

from utils import fast_load_obj, plot_hand_w_object

def Visualize():
    grasps = glob.glob('data/grasps/obj_*')
    grasps.sort()

    from manopth.manolayer import ManoLayer
    mano_layer_right = ManoLayer(
            mano_root='/home/ecorona/hand_grasppoint_gan/manopth/mano/models/', side='right', use_pca=True, ncomps=45, flat_hand_mean=True)

    for i, grasp in tqdm.tqdm(enumerate(grasps)):
        with open(grasp, 'rb') as f:
            hand = pickle.load(f, encoding='latin')

        filename = 'data/models/' + hand['body'][36:]

        objname = str.split(filename, 'nontextured_transformed.wrl')[0] + 'textured.obj'
        obj = fast_load_obj(open(objname, 'rb'))[0]

        obj_verts = obj['vertices']
        obj_faces = obj['faces']

        mano_trans = hand['mano_trans']
        posesnew = np.concatenate(([hand['pca_manorot']], hand['pca_poses']), 1)

        hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(posesnew), th_trans=torch.FloatTensor(mano_trans))
        hand_vertices = hand_vertices.cpu().data.numpy()[0]/1000
        hand_faces = mano_layer_right.th_faces.cpu().data.numpy()

        plot_hand_w_object(obj_verts, obj_faces, hand_vertices, hand_faces)

if __name__ == '__main__':
    Visualize()
