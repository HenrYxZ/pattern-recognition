import glob
import train
import test
import utils
import numpy as np
from time import time
from model import Model
from vlad import Vlad

def main():
    # img_files = ['oxbuild-images/all_souls_000000.jpg', ...]
    img_files = glob.glob("oxbuild-images/*.jpg")
    # img_names = ['all_souls_000000', ...]
    img_names = utils.get_images_names(img_files)
    # query_indices = [11, 21, ...]
    query_indices = utils.get_queries(img_names)
    files_for_codebook = img_files
    for index in query_indices:
        del files_for_codebook[index]

    # Training
    #---------------------------------------------------------------------------
    # Extracting descriptors
    #
    # start = time()
    # descriptors_count = train.calculate_descriptors(files_for_codebook)
    # end = time()
    # elapsed_time = utils.humanize_time(end - start)
    # print("Elapsed time getting the descriptors {0}.".format(elapsed_time))
    #
    # #Get the sample of 100k descriptors
    # start = time()
    # sample = train.get_sample()
    # end = time()
    # elapsed_time = utils.humanize_time(end - start)
    # print("Elapsed time getting the sample {0}.".format(elapsed_time))
    #
    # # Clustering
    #
    # k = 64
    # start = time()
    # clusters = train.get_clusters(k, sample)
    # end = time()
    # elapsed_time = utils.humanize_time(end - start)
    # print("Elapsed time clustering for k={0} {1}".format(k, elapsed_time))
    # np.savetxt("clusters64.csv", clusters, delimiter=",")
    #
    # k = 128
    # start = time()
    # clusters = train.get_clusters(k, sample)
    # end = time()
    # elapsed_time = utils.humanize_time(end - start)
    # print("Elapsed time clustering for k={0} {1}".format(k, elapsed_time))
    # np.savetxt("clusters128.csv", clusters, delimiter=",")
    #
    # k = 256
    # start = time()
    # clusters = train.get_clusters(k, sample)
    # end = time()
    # elapsed_time = utils.humanize_time(end - start)
    # print("Elapsed time clustering for k={0} {1}".format(k, elapsed_time))
    # np.savetxt("clusters256.csv", clusters, delimiter=",")


    # Vlad
    clusters = np.loadtxt("clusters128.csv", delimiter=",")
    vlad = Vlad(clusters, 64)

    vlad_matrix = None
    i = 0
    start = time()
    for image_path in files_for_codebook:
        print(str(i) +  "/" + str(len(files_for_codebook)))
        descriptors = train.get_descriptor_from_image_path(image_path)
        vlad_imagen = vlad.get_image_vlad(descriptors)
        if vlad_matrix is None:
            vlad_matrix = vlad_imagen
        else:
            vlad_matrix = np.vstack((vlad_matrix,vlad_imagen))

        i += 1
    end = time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time vlad {0}.".format(elapsed_time))
    np.savetxt("vlad128.csv",vlad_matrix,delimiter=",")






    # Testing
    #---------------------------------------------------------------------------

    # Hacer queries

if __name__ == '__main__':
	main()