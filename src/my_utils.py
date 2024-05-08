# author: Peter Kovac, xkovac66

from pathlib import Path
import os
import sys; sys.path.append(Path(os.getcwd()).parent.absolute().__str__());
import tensorflow as tf
from typing import Dict

import dlib
import cv2
from matplotlib import pyplot as plt
import numpy as np

from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from tf_smpl.batch_smpl import SMPL

from keras.losses import MeanSquaredError
import keras


def extract_flame_params(data: tf.Tensor) -> Dict[str, tf.Tensor]:
    params = {
        'trans': tf.cast(data[..., :3    ], tf.float64),
        'rot'  : tf.cast(data[..., 3:6   ], tf.float64),
        'pose' : tf.cast(data[..., 6:18  ], tf.float64),
        'shape': tf.cast(data[..., 18:318], tf.float64),
        'exp'  : tf.cast(data[..., 318:  ], tf.float64)
    }

    return params

def detect_face_landmarks_dlib(image: np.array, show: bool=False):
    if not type(image) == np.ndarray:
        image = image.numpy()
    
    if image.shape[0] == 1:
        image = image[0]
    
    if not image.mean() > 1:
        image = (image * 255)
    
    image = image.astype(np.uint8)

    print(image.shape, type(image))
    predictor_path = "./data/shape_predictor_68_face_landmarks.dat"  # Path to the downloaded model
    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor(predictor_path)  # Facial landmarks predictor
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    faces = detector(image_rgb, 1)  # Detect faces

    print(faces, type(faces))

    face = faces[0]

    landmarks = predictor(image_rgb, face)  # Predict landmarks
    landmarks_points = []
    for n in range(0, 68):  # There are 68 landmark points
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
        # Draw the landmark points

    landmarks_points = np.array(landmarks_points)

    if not show:
        return landmarks_points

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(image_rgb)
    ax.scatter(*landmarks_points.T, s=4, color='r')
    ax.axis('off')
    ax.set_title("Facial Landmarks Detected by Dlib")
    plt.show()

    return landmarks_points



def detect_face_landmarks_tf(image: np.array, landmarks_model, show=False) -> np.array:
    original_width = image.shape[1]
    original_height = image.shape[0]

    if not type(image) == np.ndarray:
        image = np.array(image)

    if image.shape[0] == 1:
        image = image[0]

    if image.shape[0] != 512 or image.shape[1] != 512:
        image = tf.image.resize(image, (512, 512)).numpy()
    
    if image.mean() > 1:
        image = image / 255.

    # Detect the face landmarks

    image_input = tf.expand_dims(image, 0)

    output = landmarks_model(image_input)
    xs = []
    ys = []

    h, w, _ = image.shape

    for i in range(0, len(output[0]), 2):
        x = output[0][i] * w
        y = output[0][i + 1] * h
        

        xs.append(x)
        ys.append(y)

    landmarks = np.array(list(zip(xs, ys)))

    # Calculate the scale factors
    scale_x = original_width / 512
    scale_y = original_height / 512

    # Scale the landmarks back to the original image size
    reshaped_landmarks = np.array([landmarks[:, 0] * scale_x, landmarks[:, 1] * scale_y])
    if not show:
        return reshaped_landmarks

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(image)
        ax.scatter(*landmarks.T, s=4, color='r')
        ax.axis('off')
        ax.set_title("Facial Landmarks Detected by TensorFlow")
        plt.show()
    
    return reshaped_landmarks

import torch
import open3d as o3d

# from pytorch3d.structures import Meshes

# from pytorch3d.renderer import (
#     look_at_view_transform,
#     FoVPerspectiveCameras,
#     PointLights,
#     RasterizationSettings,
#     MeshRenderer,
#     MeshRasterizer,
#     SoftPhongShader,
#     HardPhongShader,
#     TexturesVertex
# )

class TexturesVertex:
    def __init__(verts_features=None):
        pass
    


# def render_mesh(ply_path: str) -> np.array:
#     o3d_mesh = o3d.io.read_triangle_mesh(ply_path)
#     o3d_mesh.compute_vertex_normals()

#     # Step 2: Convert Open3D mesh to PyTorch3D mesh
#     verts = torch.tensor(np.asarray(o3d_mesh.vertices), dtype=torch.float32)
#     faces = torch.tensor(np.asarray(o3d_mesh.triangles), dtype=torch.int64)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Define a simple color for the vertices (uniform gray in this example)
#     # verts_rgb = torch.full([1, verts.shape[0], 3], 0.5, dtype=torch.float32).to(device)  # Gray color
#     verts_rgb = torch.full((1, verts.shape[0], 3), 0.8, dtype=torch.float32).to(device)  # Uniform skin-like color
#     verts_rgb[:, :, 0] = 1.0  # Red channel
#     verts_rgb[:, :, 2] = 0.6  # Blue channel


#     # Create a Textures object with vertex colors
#     textures = TexturesVertex(verts_features=verts_rgb)

#     # Create a Meshes object for PyTorch3D
#     mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

#     # Step 3: Set up the renderer
#     R, T = look_at_view_transform(dist=0.6, elev=0, azim=0)
#     cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

#     # lights = PointLights(
#     #     # device=device,
#     #     location=[[0.0, 2.0, -2.0], [2.0, 1.0, 2.0], [-2.0, 1.0, 2.0]],
#     #     # diffuse_color=((0.8, 0.8, 0.8), (0.6, 0.6, 0.6), (0.6, 0.6, 0.6)),
#     #     # specular_color=((0.2, 0.2, 0.2), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1))
#     # )

#     lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
#     raster_settings = RasterizationSettings(image_size=512)


#     raster_settings = RasterizationSettings(image_size=512)

#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
#         shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
#     )

#     # Step 4: Render the mesh
#     images = renderer(mesh)

#     return images

#     # Display the rendered image using matplotlib
#     from matplotlib import pyplot as plt
#     plt.figure(figsize=(10, 10))
#     plt.imshow(images[0, ..., :3].cpu().numpy())
#     # plt.axis("off")
#     plt.show()

#returns the left and right bounds as ratios of the image width
def get_vertical_bounds(image: np.array) -> tuple[float, float]:
    total_width = int(image.shape[1])

    first_ratio, second_ratio = None, None

    for column in range(total_width):
        mean = image[:, column].mean()
        if int(mean) < 253:
            first_ratio = column / total_width
            break

    for column in range(total_width // 2, total_width):
        mean = image[:, column].mean()
        if int(mean) > 253:
            second_ratio = column / total_width
            break
    
    return first_ratio, second_ratio

# returns the top and bottom bounds as ratios of the image height
def get_horizontal_bounds(image: np.array, top_only=True) -> tuple[float, float]:
    total_height = int(image.shape[0])

    first_ratio, second_ratio = None, None

    for row in range(total_height):
        mean = image[row, :].mean()
        if int(mean) < 253:
            first_ratio = row / total_height
            break
    
    if top_only:
        return first_ratio, None

    for row in range(total_height // 2, total_height):
        mean = image[row, :].mean()
        if int(mean) > 253:
            second_ratio = row / total_height
            break
    
    return first_ratio, second_ratio

def crop_image(image: np.array, top_only=False) -> np.array:
    width = image.shape[1]
    height = image.shape[0]

    h_bound1, h_bound2 = get_vertical_bounds(image)
    h_crop1 = int(h_bound1 * width)
    h_crop2 = int(h_bound2 * width)

    h_crop1 = max(0, min(h_crop1, width - 1))
    h_crop2 = max(0, min(h_crop2, width))

    v_bound1, v_bound2 = get_horizontal_bounds(image, top_only=top_only)

    v_crop1 = int(v_bound1 * height)
    v_crop1 = max(0, min(v_crop1, height))

    if top_only:
        v_crop2 = height
    else:
        v_crop2 = int(v_bound2 * height)
        v_crop2 = max(0, min(v_crop2, height))

    cropped_image = image[v_crop1:v_crop2, h_crop1:h_crop2]
    # output_image = tf.image.resize(cropped_image, (224, 224))

    return cropped_image

def render_face_from_flame(flame_params: Dict[str, tf.Tensor], smpl):
    tf_trans = flame_params['trans']
    tf_shape = flame_params['shape']
    tf_exp = flame_params['exp']
    tf_rot = flame_params['rot']
    tf_pose = flame_params['pose']

    flame_model = tf.squeeze(smpl(tf_trans, tf.concat([tf_shape, tf_exp], axis=-1), tf.concat([tf_rot, tf_pose], axis=-1)))

    vertices = tf.cast(flame_model, tf.float32)
    faces = smpl.f

    rendered_image = render_image_from_mesh(vertices, faces)

    return rendered_image, (vertices, faces)

import torch.nn as nn

def preprocess_output(smpl, output):
    horizontal_crop = 150
    flame_parameters = extract_flame_params(output)
    rendered_face = render_face_from_flame(flame_parameters, smpl)

    # Keep all operations in TensorFlow
    output_image = tf.slice(rendered_face, [0, horizontal_crop, 0], [-1, rendered_face.shape[1] - 2 * horizontal_crop, 3])
    output_image = output_image * 255
    output_image = tf.cast(output_image, tf.uint8)

    # Calculate cropping to square the image
    width = tf.shape(output_image)[1]
    height = tf.shape(output_image)[0]

    horizontal_crop = np.abs(((width - height) // 2).numpy().item())

    output_image = output_image[horizontal_crop:-horizontal_crop, :]

    # Resize maintaining the gradient graph
    output_image = tf.image.resize(output_image, (224, 224))
    
    return output_image

def train(model, dataset, device: str = 'GPU:0', epochs: int = 1):
    mse_loss = MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()  # Add an optimizer

    smpl_model_name = '../models/generic_model.pkl'
    smpl = SMPL(smpl_model_name)

    landmarks_model_path = os.path.join("../Human-Face-Landmark-Detection-in-TensorFlow", "files", "model.h5")
    landmarks_model = keras.models.load_model(landmarks_model_path)
    landmarks_model.trainable = False

    vertical_offset = 25

    with tf.device(device):
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for input_img, input_landmarks in zip(dataset['images'], dataset['landmarks']):
                images = tf.expand_dims(input_img, 0)
                with tf.GradientTape() as tape: 
                    output = model(images, training=True)
                    output_img = preprocess_output(smpl, output)
                    output_landmarks = detect_face_landmarks_tf(output_img, landmarks_model)

                    loss = mse_loss(input_landmarks, output_landmarks)

                    print(f"Loss: {loss}")

                    print(loss, model.trainable_variables)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    return gradients

                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")


def get_lmks_to_1D(lmks_model, input_image):
    if type(input_image) == list:
        input_image = tf.convert_to_tensor()

    if input_image.shape[0] != 1:
        input_image = tf.expand_dims(input_image, 0)

    if input_image.shape[-1] == 4:
        input_image = input_image[:, :, :, :3]

    return lmks_model(tf.image.resize(input_image, (512, 512)))[0]

def get_lmks_to_2D(lmks_model, input_image):
    if type(input_image) == list:
        input_image = tf.convert_to_tensor()

    if input_image.shape[0] != 1:
        input_image = tf.expand_dims(input_image, 0)

    return lmks_model(tf.image.resize(input_image, (512, 512)))[0]
