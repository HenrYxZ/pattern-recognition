__author__ = 'lucas'

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
import numpy as np
from vlad import Vlad
import time
import descriptors
import matplotlib.pyplot as plt
import utils
import pickle
from dataset import Dataset



#returns (X,y) tuple where X is the feature matrix and y is the class vector
def process_dataset(codebook, set_paths):
    vlad = Vlad(codebook)
    X = None
    y = []
    i = 0
    for class_paths in set_paths:
        for img_path in class_paths:
            keypoints, image_descriptors = descriptors.sift(cv2.imread(img_path.replace('\\','/')))
            image_vlad = vlad.get_image_vlad(image_descriptors)
            if X == None:
                X = image_vlad
            else:
                X = np.vstack((X, image_vlad))
        y += len(class_paths)*[i]
        i += 1
    return X, y

def svm_train(codebook, training_set_paths):
    X, y = process_dataset(codebook, training_set_paths)
    svm = svm_optimize(X,y)
    return svm


def svm_optimize(X, y):
    best_i = -10
    best_j = -10
    best_svm = None
    best_accuracy = 0
    for i in range(-10,11):
        for j in range(-10,11):
            c_value = 2**i
            gamma_value = 2**j
            clf = svm.SVC(C=c_value, gamma=gamma_value)
            scores = cross_validation.cross_val_score(clf, X, y, cv=5)
            accuracy = scores.mean()
            if accuracy > best_accuracy:
                best_i = i
                best_j = j
                best_svm = clf
    print("Parameter optimization:")
    print("C = 2^{0}".format(best_i))
    print("gamma = 2^{0}".format(best_j))
    print("Accurracy = {0}".format(best_accuracy))
    return best_svm


def svm_test(svm, codebook, testing_set_paths):
    X, y_real = process_dataset(codebook, testing_set_paths)
    y_predicted = svm.predict(X)
    cm = confusion_matrix(y_real, y_predicted)
    print("confusion matrix")
    print(cm)
    accuracy = accuracy_score(y_real, y_predicted)
    print("Test Acuracy {0}".format(accuracy))
    plot_confusion_matrix(cm, "confusion matrix (accuracy {0})".format(accuracy))


def generate_codebook(k, sample):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sample)
    return kmeans.cluster_centers_


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)


def main():
    # sample = descriptors.random_sample(100000)
    # print("sample obtenido")
    # np.savetxt("sample100k.csv", sample, delimiter=',')
    #sample = np.loadtxt("sample100k.csv", delimiter=',')
    # print(sample.shape)
    k = 64
    #clustering
    #print("Haciendo clustering...")
    #codebook = generate_codebook(k, sample)
    #np.savetxt("codebook60",codebook,delimiter=',')
    codebook = np.loadtxt("codebook60", delimiter=',')
    dataset = pickle.load(open('dataset.obj', "rb"))
    #train
    print("Entrenando...")
    start = time.time()
    svm = svm_train(codebook, dataset.get_train_set())
    end = time.time()
    s = "Elapsed time training {0}".format(utils.humanize_time(end - start))
    print(s)
    #test
    print("Testeando...")
    start = time.time()
    svm_test(svm, codebook, dataset.get_test_set())
    end = time.time()
    s = "Elapsed time training {0}".format(utils.humanize_time(end - start))
    print(s)

if __name__ == "__main__":
    main()


