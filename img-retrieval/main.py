import glob
import cv2

def main():
	img_files = glob.glob("oxbuild-images/*.jpg")
	
	# Training
	#---------------------------------------------------------------------------
	
	descriptors = train.get_descriptors(img_files)
	clusters = train.get_clusters(descriptors)
	
	# Testing
	#---------------------------------------------------------------------------

	# Hacer queries

if __name__ == '__main__':
	main()