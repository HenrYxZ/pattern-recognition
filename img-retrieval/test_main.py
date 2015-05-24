__author__ = 'lucas'

import numpy as np
import utils
import glob
from test import Test
from test import Query

def main():
    img_files = glob.glob("oxbuild-images/*.jpg")
    # img_names = ['all_souls_000000', ...]
    # query_indices = [11, 21, ...]
    img_names = utils.get_images_names(img_files)
    query_indices = utils.get_queries(img_names)
    files_for_codebook = img_files
    for index in query_indices:
        del files_for_codebook[index]
        del img_names[index]
    img_names = utils.get_images_names(img_files)

    land_marks = ['all_souls','ashmolean','balliol','bodleian','christ_church','cornmarket','hertford','keble','magdalen','pitt_rivers','radcliffe_camera']

#64
    # print("leyendo codebook...")
    # codebook = np.loadtxt("clusters64.csv", delimiter=",")
    # print("leyendo vlad matrix...")
    # vlad = np.loadtxt("vlad64.csv", delimiter=",")
    # print("listo")
    # test = Test(vlad,codebook,img_names,"euclidean")
    # precisions = []
    # for lm in land_marks:
    #     for i in range(5):
    #         index = str(i+1)
    #         precision = test.do_query(lm + "_" + index)
    #         precisions.append(precision)
    # print("64 euclidean map = "),
    # print(np.average(precisions))
    #
    # test = Test(vlad,codebook,img_names,"hellinger")
    # precisions = []
    # for lm in land_marks:
    #     for i in range(5):
    #         index = str(i+1)
    #         precision = test.do_query(lm + "_" + index)
    #         precisions.append(precision)
    # print("64 hellinger map = "),
    # print(np.average(precisions))

#128


#256
    # print("leyendo codebook...")
    # codebook = np.loadtxt("clusters256.csv", delimiter=",")
    # print("leyendo vlad matrix...")
    # vlad = np.loadtxt("vlad256.csv", delimiter=",")
    # print("listo")
    # test = Test(vlad,codebook,img_names,"euclidean")
    # precisions = []
    # for lm in land_marks:
    #     for i in range(5):
    #         index = str(i+1)
    #         precision = test.do_query(lm + "_" + index)
    #         precisions.append(precision)
    # print("256 euclidean map = "),
    # print(np.average(precisions))
    #
    # test = Test(vlad,codebook,img_names,"hellinger")
    # precisions = []
    # for lm in land_marks:
    #     for i in range(5):
    #         index = str(i+1)
    #         precision = test.do_query(lm + "_" + index)
    #         precisions.append(precision)
    # print("256 hellinger map = "),
    # print(np.average(precisions))


if __name__ == "__main__":
    main()