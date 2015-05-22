import cv2
import glob

def get_distances(current_index, model, method="euclidean"):
	''' Calculates the distances between the vlad vector in the current index
	and the vlad of every other image. 

	Args:
		current_index (int): The index for the current image.
		model (Model): The object that contains the vlads of the images.

	Returns:
		numpy array of floats: The distances to each other image vlad's. 
	'''
	pass

def get_descriptors(gray_img, resize=0):
	''' Gets a list of 128 - dimensional descriptors using
	SIFT and DoG for keypoints.

	Args:
		gray_img (grayscale matrix): The grayscale image that will be used.

	Returns:
		list of floats array: The descriptors found in the image.
	'''
	if resize > 0:
		h, w = gray_img.shape
		if h > resize or w > resize:
			if h > w:
				new_h = 640
				new_w = (640 * w) / h
			else:
				new_h = (640 * h) / w
				new_w = 640
			gray_img = cv2.resize(gray_img, (new_h, new_w))
	sift = cv2.SIFT()
	kp, des = sift.detectAndCompute(gray_img, None)
	return des

def get_images_names(img_files):
	''' Gets the names for the images in order. For example "all_souls_000013".

	Args:
		img_files (list of strings): The paths for every image in the dataset.

	Returns:
		list of strings: The names without the folder path and the extension.
	'''
	if "/" in img_files[0]:
		separator = "/"
	else:
		separator = "\\"
	return [path.split(separator)[-1].split(".")[0] for path in img_files]

def get_queries(img_names):
	''' Gets the 55 query image names that are included in the dataset.

	Args:
		img_names (list of strings): The names of the images in order.

	Returns:
		list of int: indices of the query images.
	'''
	filenames = glob.glob("groundtruth/*_query.txt")
	indices = []
	for fname in filenames:
		with open(fname) as f:
			name = f.readline().split(" ")[0]
		# Because names are in the format "oxc1_name"
		name = name[5:]
		indices.append(img_names.index(name))
	return indices

def read_desc_files(sample):
	descriptors = []
	sample.sort()
	desc_files = glob.glob("desc_*")
	separators = [fname.split(".")[0].split("_")[-1] for fname in desc_files]
	indices = [] * len(desc_files)
	counter = 0
	for index in sample:
		if index <= separators[counter]:
			indices[counter].append(index)
		else:
			counter += 1
			indices[counter].append(index)
	for i in range(len(indices)):
		if len(indices[i]) > 0:
			this_descs = pickle.load(open(desc_files[i], "rb"))
			for index in indices[i]:
				descriptors.append(this_descs[index])
	return np.array(descriptors)

def humanize_time(secs):
	'''
	Extracted from http://testingreflections.com/node/6534
	'''
	mins, secs = divmod(secs, 60)
	hours, mins = divmod(mins, 60)
	return '%02d:%02d:%02d' % (hours, mins, secs)