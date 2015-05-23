import cv2
import utils
import numpy as np
import cPickle as pickle
import glob
from sklearn.cluster import KMeans


def get_descriptor_from_image_path(path):
    resize_to = 640
    gray_img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return utils.get_descriptors(gray_img, resize=resize_to)


def calculate_descriptors(img_files):
	''' Gets the descriptors for every image in the input. Stores them in files.

	Args:
		img_files (list of string): The path for each image file in the dataset.

	Returns:
		int: number of descriptors.
	'''
	descriptors = None
	files_count = len(img_files)
	step = (5 * files_count) / 100
	max_size = 250
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
		# Stores the descriptors in a file to avoid memory errors
		if i % max_size == 0 and i > 0:
			max_desc_ind = descriptors_count - 1
			storage_count = i / max_size
			storage_name = "des_{0}_{1}_{2}.np".format(
				storage_count, min_desc_ind, max_desc_ind
			)
			pickle.dump(descriptors, open(storage_name, "wb"), protocol=2)
			min_desc_ind += len(descriptors)
			descriptors = None
	return descriptors_count

def get_sample():
	# Sacar random sample de 100k
	des_files = glob.glob("des_*")
	file_indices = [int(des_f.split("_")[1]) for des_f in des_files]
	max_index = 0
	max_value = 0
	for i in range(len(file_indices)):
		current_value = file_indices[i]
		if current_value > max_value:
			max_value = current_value
			max_index = i
	last_file = des_files[max_index]
	print ("Last file is: {0}".format(last_file))
	des_count = int(last_file.split(".")[0].split("_")[-1])
	print ("Descriptors count is: {0}".format(des_count))
	sample_indices = np.arange(des_count)
	sample_size = 100000
	np.random.shuffle(sample_indices)
	sample = utils.read_des_files(sample_indices[:sample_size])
	pickle.dump(sample, open("sample.np", "wb"), protocol=2)
	print("Sample of shape: {0}".format(sample.shape))
	return sample

def get_clusters(k, sample):
	''' Calculates the k clusters centers which are going to be the codewords
	for our codebook. It only uses a random sample of 100k of the descriptors
	and applies the k-means clustering algorithm to them.

	Args:
		k (int): The number of clusters.
		sample (numpy matrix of float32): The 100k descriptors sample.

	Returns:
		list of floats array: Each array is a cluster mean vector (D = 128).
	'''
	# Clusterizar
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(sample)
	return kmeans.cluster_centers_
