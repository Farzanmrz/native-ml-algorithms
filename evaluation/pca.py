import os
import numpy as np
import matplotlib.pyplot as plt
from models import pca
from utilities import load_yale_faces
from PIL import Image
import cv2
import random


def main():
	"""
    Main function to perform PCA on Yale Faces dataset and generate a reconstruction video.
    """
	# Set the directory containing the Yale Faces dataset
	data_dir = "../data/yalefaces"

	# Set the output directory for generated files
	output_dir = "../data/outputs"

	# Ensure the output directory exists
	os.makedirs(output_dir, exist_ok = True)

	# Set the image size for resizing
	img_size = (40, 40)

	# Load the Yale Faces dataset
	data = load_yale_faces(data_dir, img_size)

	# Perform PCA on the dataset
	z, w, vt = pca(data, n_components = 2)

	# Plot the first two principal components
	plt.scatter(z[ :, 0 ], z[ :, 1 ])
	plt.title('PCA of Yale Faces')
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.show()

	# Select a random subject image file
	random_image_file = random.choice(os.listdir(data_dir))

	# Open the selected image, resize it, and flatten it
	centered_image = np.array(Image.open(os.path.join(data_dir, random_image_file)).resize(img_size)).ravel().reshape(1, -1)

	# Convert centered_image to float64 for subtraction
	centered_image = centered_image.astype(np.float64)

	# Center the flattened image by subtracting the mean of the data matrix
	centered_image -= np.mean(data, axis = 0)

	# Generate the reconstruction video
	frames = [ np.uint8((np.dot((centered_image @ vt[ :k, : ].T), vt[ :k, : ]) + np.mean(data, axis = 0)).reshape(img_size)) for k in range(1, len(vt) + 1) ]

	# Set the output video file path
	output_video_path = os.path.join(output_dir, 'reconstruction.avi')

	# Initialize the video writer object with MJPG codec at 200 fps
	out_vid = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 200.0, img_size)

	# Convert each frame to BGR format and write it to the video
	[ out_vid.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)) for frame in frames ]

	# Release the video writer object
	out_vid.release()


if __name__ == "__main__":
	# Execute the main function
	main()
