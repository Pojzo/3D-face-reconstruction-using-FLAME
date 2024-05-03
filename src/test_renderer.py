import tensorflow as tf
import tensorflow_graphics as tfg

from tensorflow_graphics.rendering.camera import orthographic

def main():
    # Example 3D points
    points_3d = tf.constant([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=tf.float32)  # Shape: [3, 3]

    # Camera focal length (for x and y axes)
    focal_length = tf.constant([
        [1000.0, 1000.0],
        [1000.0, 1000.0],
        [1000.0, 1000.0]
    ], dtype=tf.float32)  # Shape: [3, 2]

    # Principal point (typically the image center)
    principal_point = tf.constant([
        [640.0, 480.0],
        [640.0, 480.0],
        [640.0, 480.0]
    ], dtype=tf.float32)  # Shape: [3, 2]

    # Projecting 3D points onto the 2D plane
    points_2d = orthographic.project(
        points_3d,
    )

    return points_2d

    # Run and print the projected 2D points
    with tf.compat.v1.Session() as sess:
        projected_points = sess.run(points_2d)
        print("Projected 2D Points:\n", projected_points)

points = main()