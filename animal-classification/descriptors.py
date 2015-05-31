import cv2

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
			img = cv2.resize_to(img, (new_h, new_w))
	sift = cv2.SIFT()
	kp, des = sift.detectAndCompute(img, None)
	return des