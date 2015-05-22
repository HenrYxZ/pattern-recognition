import cv2
import utils
import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans

def calculate_descriptors(img_files):
	''' Gets the descriptors for every image in the input. Stores them in files.

	Args:
		img_files (list of string): The path for each image file in the dataset.

	Returns:
		int: number of descriptors.
	'''
	descriptors = []
	files_count = len(img_files)
	step = (5 * files_count) / 100
	max_size = 300
	descriptors_count = 0
	min_desc_ind = 0
	max_desc_ind = 0
	resize_to = 640
	for i in range(files_count):
		# Get the descriptors for each grayscale image 
		filename = img_files[i]
		gray_img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		current_img_des = utils.get_descriptors(gray_img, resize=resize_to)
		descriptors_count += len(current_img_des)
		descriptors.append(current_img_des)
		if i % step == 0:
			percentage = (i * 100) / files_count
			print(
				"Getting descriptors in image number {0} of {1} ({2}%)".format(
					i, files_count, percentage
				)
			)
		# Stores the descriptors in a file to avoid memory errors
		if i % max_size == 0 and i > 299:
			max_desc_ind = len(descriptors) - 1
			storage_name = "des_{0}_{1}.np".format(min_desc_ind, max_desc_ind)
			pickle.dump(descriptors, open(storage_name, "wb"))
			min_desc_ind = len(descriptors)
			descriptors = []
	return descriptors_count

def get_clusters(k):
	''' Calculates the k clusters centers which are going to be the codewords
	for our codebook. It only uses a random sample of 100k of the descriptors
	and applies the k-means clustering algorithm to them.

	Args:
		k (int): The number of clusters.

	Returns:
		list of floats array: Each array is a cluster mean vector (D = 128).
	'''
	# Sacar random sample de 100k
	n_rows = matrix.shape[0]
	sample_size = 100000
	sample_indices = np.random.choice(n_rows, sample_size)
	sample = utils.read_desc_files(sample_indices)
	print("Sample of shape: {0}".format(sample.shape))
	# Clusterizar
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(sample)
	return kmeans.cluster_centers_
