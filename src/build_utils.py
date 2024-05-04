# author: Peter Kovac, xkovac66

import tensorflow as tf
import keras
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
from my_utils import detect_face_landmarks_tf

class OutputLayer(layers.Layer):
    def __init__(self):
        super(OutputLayer, self).__init__()
        # adapted from the TF_FLAME repository
        # https://github.com/TimoBolkart/TF_FLAME
        self.tf_trans = self.add_weight(shape=(1, 3), initializer="zeros", dtype=tf.float64, trainable=False)
        self.tf_rot = self.add_weight(shape=(1, 3), initializer="zeros", dtype=tf.float64, trainable=False)
        self.tf_pose = self.add_weight(shape=(1, 12), initializer="zeros", dtype=tf.float64, trainable=True)
        self.tf_shape = self.add_weight(shape=(1, 300), initializer="zeros", dtype=tf.float64, trainable=True)
        self.tf_exp = self.add_weight(shape=(1, 100), initializer="zeros", dtype=tf.float64, trainable=True)

    def call(self, inputs):
        # Concatenate all parameters into one vector
        return tf.concat([self.tf_trans, self.tf_rot, self.tf_pose, self.tf_shape, self.tf_exp], axis=1)

def create_transform():
    transform = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    return transform

from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import Input

def create_model():
    # Define the input shape
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape)

    # Flatten the input
    flattened_inputs = Flatten()(inputs)

    # Add a fully connected layer
    fc1 = Dense(64, activation='relu')(flattened_inputs)

    # Add another fully connected layer
    fc2 = Dense(32, activation='relu')(fc1)

    # Define the output layer
    output_layer = OutputLayer()

    # Connect the second fully connected layer to the output layer
    outputs = output_layer(fc2)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

def create_dataset(dataset_path: str, landmarks_model_path, transform: ImageDataGenerator=None, image_size: tuple = (224, 224)) -> tf.Tensor:
    landmarks_model = keras.models.load_model(landmarks_model_path)
    vertical_crop = (450 - 316) // 2

    images = { "images": [], "landmarks": []}
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(dataset_path, filename)
            img = load_img(img_path)
            img_array = img_to_array(img)
            img_array =  img_array[:, vertical_crop:-vertical_crop, :]
            landmarks = landmarks_model(tf.expand_dims(tf.image.resize(img_array, (512, 512)), 0))
            # landmarks = detect_face_landmarks_tf(img_array, landmarks_model)
            images["images"].append(tf.convert_to_tensor(img_array) / 255.)
            images["landmarks"].append(tf.convert_to_tensor(landmarks))
            print(f"Loaded image: {filename}, len(images): {len(images['images'])}")
            if len(images["images"]) == 20:
                break

    del landmarks_model 
    return images
