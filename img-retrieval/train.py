import cv2
import utils
import numpy as np
from sklearn.cluster import KMeans

def get_descriptors(img_files):
	''' Gets the descriptors for every image in the list and concatenates them.

	Args:
		img_files (list of string): The path for each image file in the dataset.

	Returns:
		numpy matrix of floats: Each row is a 128 dimensional descriptor.
	'''
	descriptors = None
	files_count = len(img_files)
	for i in range(files_count):
		filename = img_files[i]
		gray_img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		current_img_des = utils.get_descriptors(gray_img)
		if descriptors is None:
			descriptors = current_img_des
		else:
			descriptors = np.vstack((descriptors, current_img_des))
		step = (5 * files_count) / 100
		if i % step == 0:
			percentage = (i * 100) / files_count
			print(
				"Getting descriptors in image number {0} of {1} ({2}%)".format(
					i, files_count, percentage
				)
			)
	print("Descriptors of shape: {0}".format(descriptors.shape))
	return descriptors

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
	indices = np.arange(len(descriptors))
	np.random.shuffle(indices)
	sample_size = 100000
	indices = indices[:sample_size]
	sample = descriptors[indices[0]]
	for i in range(1, sample_size):
		index = indices[i]
		sample = np.vstack((sample, descriptors[index]))
	print("Sample of shape: {0}".format(sample.shape))
	# Clusterizar
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(sample)
	return kmeans.cluster_centers_
