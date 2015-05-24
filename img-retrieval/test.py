import utils
import train
from vlad import Vlad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Test:
    def __init__(self, vlad_matrix, codebook, files_names, distance_type):
        self.vlad_matrix = vlad_matrix
        self.distance_type = distance_type
        self.files_names = files_names
        self.codebook = codebook

    def do_query(self, query_name):
        query = Query(self.vlad_matrix,self.codebook,self.files_names, self.distance_type, query_name)
        #query.print_query()
        query.calculate_ranking()
        precision = query.calculate_recall_precision()
        query.plot()
        return precision


class Query:
    def __init__(self, vlad_matrix, codebook, files_names, distance_type, query_name):
        self.vlad_matrix = vlad_matrix
        self.codebook = codebook
        self.n_clusters = len(codebook)
        self.files_names = files_names
        self.distance_type = distance_type
        self.query_name = query_name
        self.query_image_name = get_query_file("groundtruth/" + self.query_name + "_query.txt")
        self.good_files_names = get_file_list("groundtruth/" + self.query_name + "_good.txt")
        ok_files_names = get_file_list("groundtruth/" + self.query_name + "_ok.txt")
        junk_files_names = get_file_list("groundtruth/" + self.query_name + "_junk.txt")
        self.bad_files_names = set(files_names) - set(self.good_files_names) - set(ok_files_names) - set(junk_files_names)
        self.good_indexes = [self.files_names.index(file_name) for file_name in self.good_files_names if file_name in self.files_names]
        self.bad_indexes = [self.files_names.index(file_name) for file_name in self.bad_files_names if file_name in self.files_names]
        self.n_good = len(self.good_indexes)
        self.sift = train.get_descriptor_from_image_path("oxbuild-images/"+ self.query_image_name)
        self.vlad = Vlad(codebook).get_image_vlad(self.sift)
        try:
            self.good_vlads = self.vlad_matrix[self.good_indexes,:]
        except IndexError:
            print("too many indexes")
            print("good indexes")
            print(self.good_indexes)

        self.bad_vlads = self.vlad_matrix[self.bad_indexes,:]


    def distance(self,v1,v2):
        if self.distance_type == "hellinger":
            sqrt_v1 = np.sign(v1) * np.sqrt(np.abs(v1))
            sqrt_v2 = np.sign(v2) * np.sqrt(np.abs(v2))
            # print("V1 = "),
            # print(v1)
            # print(v1.shape)
            # print("sqrtV1 = "),
            # print(sqrt_v1)
            # print(sqrt_v1.shape)
            return distance.euclidean(sqrt_v1,sqrt_v2)
        else:
            return distance.euclidean(v1,v2)

    def print_query(self):
        print("Query name: " + self.query_name)
        print("Query image name: " + self.query_image_name)
        print("Good names: ")
        print(self.good_files_names)
        print("Bad names: ")
        print(self.bad_files_names)
        print("Good indexes: ")
        print(self.good_indexes)
        print("Bad indexes: ")
        print(self.bad_indexes)
        print("Sift: ")
        print(self.sift)
        print("Vlad: ")
        print(self.vlad)
        print("good vlads: ")
        print(self.good_vlads)
        print("bad vlads: ")
        print(self.bad_vlads)


    def calculate_ranking(self):
        self.good_distances = []
        for good_vlad in self.good_vlads:
            self.good_distances.append(self.distance(self.vlad,good_vlad))
        self.bad_distances = []
        for bad_vlad in self.bad_vlads:
            self.bad_distances.append(self.distance(self.vlad,bad_vlad))
        self.distances = self.good_distances + self.bad_distances
        self.ranking = np.argsort(self.distances).tolist()

    def calculate_recall_precision(self):
        LR = 0.0
        R = len(self.good_distances)
        recall = []
        precision = []
        for i in range(len(self.good_distances)):
            L = i + 1
            if self.ranking.index(i) < self.n_good:
                LR += 1
            recall.append(LR / R)
            precision.append(LR / L)
        self.recall = recall
        self.precision = precision
        self.average_precision = np.average(self.precision)
        # print("precision array")
        # print(precision)
        # print("average precision")
        # print(self.average_precision)
        return self.average_precision

    def plot(self):
        plt.clf()
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.plot(self.recall,self.precision)
        plt.suptitle(self.query_name + " " + self.distance_type + " distance  (AP = " + str(self.average_precision) + ")")
        plt.savefig(self.distance_type + "_" + str(self.n_clusters) + "_" + self.query_name)


def get_query_file(fname):
    with open(fname) as f:
        name = f.readline().split(" ")[0]
        name = name[5:]
    return name + ".jpg"


def get_file_list(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines