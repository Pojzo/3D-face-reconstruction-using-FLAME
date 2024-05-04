# this code is adapted from https://github.com/szattila/pTFrenderer

import sys

sys.path.append("/content/pTFrenderer/")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.enable_eager_execution()

import open3d as o3d

from ptfrenderer import utils
from ptfrenderer import camera
from ptfrenderer import mesh

def render_image_from_mesh(vertices, faces):
    def align(xyz):
        xyz_min = tf.reduce_min(xyz, axis=[0,1], keepdims=True)
        xyz_max = tf.reduce_max(xyz, axis=[0,1], keepdims=True)
        xyz = xyz - xyz_min - (xyz_max - xyz_min) / 2.
        r = tf.sqrt(tf.reduce_max(tf.reduce_sum(xyz**2, -1)))
        xyz /= r
        angles = tf.constant([[-180., 0., 0.]]) * np.pi / 180.
        R = camera.hom(camera.euler(angles), 'R')[:,:3,:]
        xyz = camera.transform(R, xyz)
        return xyz

    def load(vertices, faces):
        xyz = tf.constant(vertices, tf.float32)[tf.newaxis,:,:]
        xyz = align(xyz)
        triangles = tf.constant(faces)
        fur_color = tf.constant([[[201., 156., 122.]]]) / 255.
        s = 0.05
        rgb = tf.random.uniform(xyz.shape) * s + (1. - s) * fur_color
        return xyz, rgb, triangles

    def light(xyz, rgb, triangles, light_direction, ambient):
        norm = mesh.vert_normals(xyz, triangles)
        lambertian_shading = tf.maximum(tf.reduce_sum(norm * light_direction, -1, keepdims=True), 0.)
        rgb_lit = rgb * (lambertian_shading + ambient)
        return rgb_lit

    import open3d as o3d

    xyz, rgb, triangles = load(vertices, faces)
    # Centralized directional light, pointing towards the origin from above
    light_direction = tf.constant([[0., -1., 0.]])
    light_direction = tf.linalg.l2_normalize(light_direction, -1)
    ambient = 0.5  # Increased ambient lighting for more even illumination
    rgb_lit = light(xyz, rgb, triangles, (1. - ambient) * light_direction, ambient)

    imsize = [256, 256]

    angle = tf.constant([[0., 0., 0.]]) * np.pi / 180.
    R = camera.hom(camera.euler(angle), 'R')
    K, T = camera.look_at_origin(5.*np.pi/180.)
    T, R = utils._broadcast(T, R)
    P = tf.matmul(T, R)

    rendered_image = mesh.render(xyz, rgb_lit, triangles, K, P, imsize)
    rendered_image = utils.alpha_matte(rendered_image, 1.0)  # white background

    return rendered_image
