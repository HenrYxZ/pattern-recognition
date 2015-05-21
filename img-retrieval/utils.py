import cv2

def get_distances(current_index, model, method="euclidean"):
	''' Calculates the distances between the vlad vector in the current index
	and the vlad of every other image. 

	Args:
		current_index (int): The index for the current image.
		model (Model): The object that contains the vlads of the images.
	'''
	pass

def get_descriptors(gray_img):
	''' Gets a list of 128 - dimensional descriptors using
	SIFT and DoG for keypoints.

	Args:
		gray_img (grayscale matrix): The grayscale image that will be used.

	Returns:
		list of floats array: The descriptors found in the image.
	'''
	sift = cv2.SIFT()
	kp, des = sift.detectAndCompute(gray_img, None)
	return des

def humanize_time(secs):
	'''
	Extracted from http://testingreflections.com/node/6534
	'''
	mins, secs = divmod(secs, 60)
	hours, mins = divmod(mins, 60)
	return '%02d:%02d:%02d' % (hours, mins, secs)