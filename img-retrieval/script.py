import glob
import utils

img_files = glob.glob("oxbuild-images/*.jpg")
img_names = utils.get_images_names(img_files)
query_indices = utils.get_queries(img_names)
files_for_codebook = img_files
for index in query_indices:
	del files_for_codebook[index]
print len(files_for_codebook)