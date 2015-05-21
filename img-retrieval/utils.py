import cv2

def get_distances(current_index, model, method="euclidean"):
	''' Calculates the distances between the vlad vector in the current index
	and the vlad of every other image. 

	Args:
		current_index (int): The index for the current image.
		model (Model): The object that contains the vlads of the images.
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

def humanize_time(secs):
	'''
	Extracted from http://testingreflections.com/node/6534
	'''
	mins, secs = divmod(secs, 60)
	hours, mins = divmod(mins, 60)
	return '%02d:%02d:%02d' % (hours, mins, secs)