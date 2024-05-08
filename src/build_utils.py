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


import tensorflow as tf
from tensorflow.keras import layers, Model

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(OutputLayer, self).__init__()
        # Adapted from the TF_FLAME repository
        # https://github.com/TimoBolkart/TF_FLAME
        self.tf_trans = self.add_weight(
            shape=(1, 3),
            initializer="zeros",
            dtype=tf.float32,
            trainable=False
        )
        self.tf_rot = self.add_weight(
            shape=(1, 3),
            initializer="zeros",
            dtype=tf.float32,
            trainable=False
        )
        self.tf_pose = self.add_weight(
            shape=(1, 12),
            initializer="zeros",
            dtype=tf.float32,
            trainable=False
        )
        self.tf_shape = self.add_weight(
            shape=(1, 300),
            initializer="zeros",
            # initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2),
            dtype=tf.float32,
            trainable=True
        )
        self.tf_exp = self.add_weight(
            shape=(1, 100),
            initializer="zeros",
            dtype=tf.float32,
            trainable=False
        )

    @tf.function
    def call(self, inputs):
        # Concatenate all parameters into one vector
        output_vector = tf.concat(
            [self.tf_trans, self.tf_rot, self.tf_pose, self.tf_shape, self.tf_exp],
            axis=1
        )
        output_vector = tf.tile(output_vector, [tf.shape(inputs)[0], 1])  # Repeat for each batch

        # Combine inputs (from dense layer) with the output_vector
        return tf.concat([inputs, output_vector], axis=1)

class FLAMEModel(Model):
    def __init__(self):
        super(FLAMEModel, self).__init__()
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg',

        )
        base_model.trainable = False
        self.encoder = Model(inputs=base_model.input, outputs=base_model.output)

        # Fully connected layers
        self.fc1 = layers.Dense(1024, activation='relu', name="dense_1", dtype=tf.float32)
        self.fc2 = layers.Dense(512, activation='relu', name="dense_2", dtype=tf.float32)

        # Custom Output Layer for FLAME parameters
        self.output_layer = OutputLayer()

    def call(self, inputs):
        features = self.encoder(inputs)
        x = self.fc1(features)
        x = self.fc2(x)
        return self.output_layer(x)

def create_optimizer():
    optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=0.01,  # Higher learning rate
    beta_1=0.9,          # Momentum (1st order)
    beta_2=0.999,        # Momentum (2nd order)
    epsilon=1e-07,
    decay=1e-4          # Learning rate decay   
    )

    return optimizer

def create_loss():
    return keras.losses.MeanSquaredError()

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