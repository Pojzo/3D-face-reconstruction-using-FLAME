'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import sys
sys.path.append(r"C:\Users\pojzi\.conda\envs\TF_FLAME\Lib\site-packages\psbody_mesh-0.4-py3.7-win-amd64.egg")

import os
import six
import argparse
import numpy as np
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from utils.landmarks import load_binary_pickle, load_embedding, tf_get_model_lmks, create_lmk_spheres

from tf_smpl.batch_smpl import SMPL

# from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from ScipyOptimizerInterface import ScipyOptimizerInterface as scipy_pt


def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ['true', 't', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'f', 'no', 'n']:
            return False
    return False

def sample_FLAME(model_fname, num_samples, out_path, visualize, sample_VOCA_template=False):
    tf_trans = tf.Variable(np.zeros((1, 3)), dtype=tf.float64, trainable=True)
    tf_rot = tf.Variable(np.zeros((1, 3)), dtype=tf.float64, trainable=True)
    tf_pose = tf.Variable(np.zeros((1, 12)), dtype=tf.float64, trainable=True)
    tf_shape = tf.Variable(np.zeros((1, 300)), dtype=tf.float64, trainable=True)
    tf_exp = tf.Variable(np.zeros((1, 100)), dtype=tf.float64, trainable=True)
    
    smpl = SMPL(model_fname)

    for i in range(num_samples):
        if sample_VOCA_template:
            tf_shape.assign(np.hstack((np.random.randn(100), np.zeros(200)))[np.newaxis, :])
            out_fname = os.path.join(out_path, f'VOCA_template_{i+1:02d}.ply')
        else:
            tf_rot.assign(np.zeros((1, 3)))
            tf_pose.assign(np.zeros((1, 12)))
            tf_shape.assign(np.hstack((np.random.randn(100), np.zeros(200)))[np.newaxis, :])
            tf_exp.assign(np.hstack((0.5 * np.random.randn(50), np.zeros(50)))[np.newaxis, :])
            out_fname = os.path.join(out_path, f'FLAME_sample_{i+1:02d}.ply')

        tf_model = tf.squeeze(smpl(tf_trans, tf.concat([tf_shape, tf_exp], axis=-1), tf.concat([tf_rot, tf_pose], axis=-1)))
        sample_mesh = Mesh(tf_model, smpl.f)  # Assuming 'smpl.f' gives the faces of the mesh

        if visualize:
            mv = MeshViewer()
            # Visualization code here; assuming some MeshViewer class or similar for visualization
            mv.set_dynamic_meshes([sample_mesh], blocking=True)
            key = input('Press (s) to save sample, any other key to continue ')
            if key.lower() == 's':
                sample_mesh.write_ply(out_fname)
        else:
            sample_mesh.write_ply(out_fname)

def main(args):
    if not os.path.exists(args.model_fname):
        print('FLAME model not found - %s' % args.model_fname)
        print(args.model_fname)
        return
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if args.option == 'sample_FLAME':
        sample_FLAME(args.model_fname, int(args.num_samples), args.out_path, str2bool(args.visualize), sample_VOCA_template=False)
    else:
        sample_FLAME(args.model_fname, int(args.num_samples), args.out_path, str2bool(args.visualize), sample_VOCA_template=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample FLAME shape space')
    parser.add_argument('--option', default='sample_FLAME', help='sample random FLAME meshes or VOCA templates')
    parser.add_argument('--model_fname', default='./models/generic_model.pkl', help='Path of the FLAME model')
    parser.add_argument('--num_samples', default='5', help='Number of samples')
    parser.add_argument('--out_path', default='./FLAME_samples', help='Output path')
    parser.add_argument('--visualize', default='True', help='Visualize fitting progress and final fitting result')
    args = parser.parse_args()
    main(args)