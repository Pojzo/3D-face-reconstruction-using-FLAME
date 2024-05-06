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
import matplotlib.pyplot as plt

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

    light_direction = tf.constant([[0., -1, -1.2]])
    light_direction = tf.linalg.l2_normalize(light_direction, -1)
    ambient = 0.6
    rgb_lit = light(xyz, rgb, triangles, (1.-ambient)*light_direction, ambient)
    imsize = [256, 256]

    angle = tf.constant([[0., 0., 0.]]) * np.pi / 180.
    R = camera.hom(camera.euler(angle), 'R')
    K, T = camera.look_at_origin(5.*np.pi/180.)
    T, R = utils._broadcast(T, R)
    P = tf.matmul(T, R)

    rendered_image = mesh.render(xyz, rgb_lit, triangles, K, P, imsize)
    rendered_image = utils.alpha_matte(rendered_image, 1.0)  # white background

    return rendered_image

def preprocess_rendered_image(image, crop=35, top_crop=5, target_size=(512, 512), pad_top=0):
    width = image[0].shape[1]
    height = image[0].shape[0]

    if pad_top != 0:
       pad = tf.ones((1, pad_top, width, image.shape[3]), dtype=image.dtype)
       image = tf.concat([pad, image], axis=1)

    cropped_image = tf.image.crop_to_bounding_box(image[:, :, :, :3], top_crop, crop // 2, height - crop + top_crop, width - crop)

    resized_image = tf.image.resize(cropped_image, target_size)

    return resized_image

def visualize_landmarks(image, landmarks, save_path=None, show_comparison=False):
    x = landmarks[::2] * image.shape[1]
    y = landmarks[1::2] * image.shape[0]
    
    if show_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        axes[1].imshow(image)
        axes[1].scatter(x, y, s=3, color='r')
        axes[1].axis('off')
        axes[1].set_title('Image with Landmarks')

    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.scatter(x, y, s=3, color='r')
        plt.title('Facial Landmarks Visualization')
        plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def visualize_input_output_landmarks(input_landmarks, output_landmarks, input_image, output_image):
    target = input_landmarks[0]
    pred = output_landmarks

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    target_x = target[::2]
    target_y = target[1::2]

    pred_x = pred[::2]
    pred_y = pred[1::2]

    axs[0][0].imshow(input_image)
    axs[0][0].scatter(target_x * input_image.shape[1], target_y * input_image.shape[0], color='orange', s=3)

    axs[0][1].imshow(output_image)
    axs[0][1].scatter(pred_x * output_image.shape[1], pred_y * output_image.shape[0], color='blue', s=3)

    plt.scatter(target_x, target_y, color='orange')
    plt.scatter(pred_x, pred_y, color='blue')
