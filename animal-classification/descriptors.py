import cv2
import glob
import cPickle as pickle
import numpy as np
import time


def sift(img):
    ''' Gets a list of 128 - dimensional descriptors using SIFT and DoG
    for keypoints and resizes the image having the larger dimension set to 640
    and keeping the size relation.

    Args:
        img (BGR matrix): The grayscale image that will be used.

    Returns:
        list of floats array: The descriptors found in the image.
    '''
    resize_to = 640
    if resize_to > 0:
        h, w, channels = img.shape
        if h > resize_to or w > resize_to:
            if h > w:
                new_h = 640
                new_w = (640 * w) / h
            else:
                new_h = (640 * h) / w
                new_w = 640
            img = cv2.resize(img, (new_w, new_h))
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def get_counts():
    # THIS FUNCTION IS VERY INEFFICIENT
    des_files = glob.glob("train/*")
    # Counts is a list with the number of descriptors for each class and the
    # last element is the total number of descriptors.
    des_files.sort()
    counts = []
    total = 0
    for des_f in des_files:
        print("reading file {0}".format(des_f))
        des = pickle.load(open(des_f, "rb"))
        print("Got descriptors of shape {0}".format(des.shape))
        total += len(des)
        counts.append(len(des))
    counts.append(total)
    return counts


def random_sample(sample_size):
    counts = [1085495, 761595, 1035142, 1195164, 721518, 950140, 1125565, 1005910, 845804, 820759, 9547092]
    total_count = counts[-1]
    counts = counts[:len(counts)-1]
    sample_indices = np.random.choice(total_count, sample_size, replace=False)
    descriptors = None
    for sample_index in sample_indices:
        descriptor = get_descriptor(sample_index, counts)
        if descriptors is None:
            descriptors = descriptor
        else:
            descriptors = np.vstack((descriptors,descriptor))
    return descriptors


def get_file_and_relative_index(absolute_index, counts):
    relative_count = 0
    file_index = 0
    # print("counts")
    # print(counts)
    for c in counts:
        if relative_count <= absolute_index < relative_count + c:
            relative_index = absolute_index - relative_count
            return file_index, relative_index
        file_index += 1
        relative_count += c
    print("FALLA!!!")
    return None

def get_descriptor(absolute_index, counts):
    file_index, relative_index = get_file_and_relative_index(absolute_index, counts)
    des_files = glob.glob("train/*")
    des_files.sort()
    des = pickle.load(open(des_files[file_index], "rb"))
    return des[relative_index, :]

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


# def random_sample(counts):
#     n = counts[-1]
#     sample_size = 100000
#     sample_indices = np.random.choice(n, sample_size, replace=False)
#     sample_indices.sort()
#     des_files = glob.glob("train/*")
#     counter = 0
#     counter_changed = False
#     sample = []
#     # This is the current descriptors list of a class
#     current_des = None
#     previous_classes_len = 0
#     for sample_index in sample_indices:
#         if sample_index < previous_classes_len + counts[counter]:
#             if current_des is None:
#                 current_des = pickle.load(open(des_files[counter], "rb"))
#         else:
#             previous_classes_len += counts[counter]
#             counter_changed = True
#             counter += 1
#             current_des = pickle.load(open(des_files[counter], "rb"))
#         # The index for the descriptor in the current class (sum the lengths
#         # of the previous classes)
#         current_class_idx = sample_index - previous_classes_len
#         if counter_changed:
#             time.sleep(2)
#             print("previous_classes_len = {0}".format(previous_classes_len))
#             print("current_class_idx = {0}".format(current_class_idx))
#             print("sample index = {0}".format(sample_index))
#             print("counter = {0}".format(counter))
#         sample.append(current_des[current_class_idx])
#     return np.array(sample)
#
#





