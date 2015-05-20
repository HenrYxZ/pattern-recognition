import cv2
import utils
import numpy as np

def get_descriptors(img_files):
	''' Gets the descriptors for every image in the list and concatenates them.

	Args:
		img_files (list of string): The path for each image file in the dataset.

	Returns:
		matrix of floats: Each column is a 128 dimensional descriptor.
	'''
	descriptors = []
	files_count = len(img_files)
	for i in range(files_count):
		filename = img_files[i]
		gray_img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		descriptors.append(utils.get_descriptors(gray_img))
		percentage = (100 * i) / files_count
		if percentage % 5 == 0:
			print ("Getting descriptors in image number {0} of {1} ({2}%).")
	return np.array(descriptors)

def get_clusters(k, descriptors):
	''' Calculates the k clusters centers which are going to be the codewords
	for our codebook. It only uses a random sample of 100k of the descriptors
	and applies the k means clustering algorithm to them.

	Args:
		k (int): The number of clusters.
		descriptors (list of floats array): The descriptors in the training set.

	Returns:
		list of floats array: Each array is a cluster mean vector (D = 128).
	'''
	# Sacar random sample de 100k
	# Clusterizar
	pass