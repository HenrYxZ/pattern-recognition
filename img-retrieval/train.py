def get_descriptors(img_files):
	''' Gets a list of 128 - dimensional descriptors for each image file using
	SIFT and DoG for keypoints.

	Args:
		img_files (list of strings): The path for each img file using glob
			package.

	Returns:
		list of floats array: The descriptors found in the image.
	'''
	# Pueden ser demasiados descriptores
	pass

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