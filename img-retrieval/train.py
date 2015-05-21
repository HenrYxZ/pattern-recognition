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
	step = (5 * files_count) / 100
	resize_to = 640
	for i in range(files_count):
		filename = img_files[i]
		is_query = True
		while is_query and i < files_count:
		gray_img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		current_img_des = utils.get_descriptors(gray_img, resize=resize_to)
		if descriptors is None:
			descriptors = current_img_des
		else:
			descriptors = np.vstack((descriptors, current_img_des))
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
	n_rows = matrix.shape[0]
	sample_size = 100000
	indices = np.random.choice(n_rows, sample_size)
	sample = descriptors[indices, :]
	print("Sample of shape: {0}".format(sample.shape))
	# Clusterizar
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(sample)
	return kmeans.cluster_centers_
