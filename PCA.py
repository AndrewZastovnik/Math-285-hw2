import numpy as np
import pickle
import pylab as plt
from NearestNeighbors import mfoldX,KNN,local_kmeans_class
from TicToc import tic,toc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class center_matrix_SVD:
    # A class to store our information about our centered matrix
    # center_matrix has 7 atributes
    # .size stores the shape of the original matrix
    # .centers stores the center of the dataset
    # a_centered is the centered original matrix
    # .U .s .V are the SVD decomposition of the centered matrix
    def __init__(self,a,dim=0):
        self.size = a.shape # Gets and stores the shape of a not sure it is really necessary
        self.centers = np.mean(a,axis=dim).reshape(1,self.size[1])
        # Reshaped as 1,n instead of ,n because that was causing problems
        self.a_centered = np.subtract(a,np.repeat(self.centers,self.size[dim],dim))
        #Creates an atribute a_centered to store the  centered a matrix
        self.U, self.s, self.V = np.linalg.svd(self.a_centered,full_matrices = False) # Runs SVD on our centered a matrix
        self.PCA = self.U@np.diagflat(self.s)
        # stores the U*S matrix in atribute PCA since s is diagnol we can just take rows
        # out of this instead of recalulating PCA for reduced dims

class mnist:
    # Creates a class that stores our mnist data. Hopefully it helps keep things more oragnized
    train_Images = pickle.load(open('mnistTrainI.p', 'rb'))
    test_Images = pickle.load(open('mnistTestI.p', 'rb'))
    train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
    test_labels = pickle.load(open('mnistTestL.p', 'rb'))


def dump_the_svd(digits):
    # Gets SVD for the training dataset
    train = center_matrix_SVD(digits.train_Images)
    pickle.dump(train,open('Training SVD Data','wb')) # dump the result to a file
    # You won't be able to load this unless you have the center_matrix_SVD class avalible
    # Not sure it really saves much time to put this into a file since svd is pretty fast

def class_error_rate(pred_labels,true_labels):
    # for calculating the error rate
    # Also returns a index vector with the position of incorrectly labeled images
    error = np.zeros(pred_labels.shape[0])
    error_index = np.zeros((pred_labels.shape[0],pred_labels.shape[1]))
    for i in range(pred_labels.shape[0]):
        error[i] = sum(pred_labels[i] != true_labels)/pred_labels.shape[1]
        # puts each
        error_index[i] = 1 - np.isclose(pred_labels[i],true_labels)
        #
    return error, error_index

def MFold_plots():
    # Mfold plots
    err = pickle.load(open('MFolderrors50','rb'))
    err2 = pickle.load(open('MFolderrors','rb'))
    err3 = pickle.load(open('MFolderrors154','rb'))
    plt.figure()
    plt.plot(np.arange(10)+1,err,color = 'Green',marker='x',markersize=10,label = '82.5%')
    plt.plot(np.arange(10)+1,err3,color = 'Yellow',marker='+',markersize=10,label = '95%')
    plt.plot(np.arange(10)+1,err2,color = 'Red',marker='o',markersize=10,label = 'Full')
    plt.grid(1)
    plt.title('Scatter plot of the three error rates')
    plt.legend(loc='upper right')
    plt.show()

def do_KNN(x,digits):
    #code to run knn for the three values of s
    test_Images_Center = np.subtract(digits.test_Images,np.repeat(x.centers,digits.test_Images.shape[0],0))
    Knn_labels, nearest = KNN(x.PCA,digits.train_Labels,test_Images_Center@np.transpose(x.V[:,:]),10)
    pickle.dump(Knn_labels,open('Knn_Full','wb'))
    Knn_labels, nearest = KNN(x.PCA[:,:154],digits.train_Labels,test_Images_Center@np.transpose(x.V[:154,:]),10)
    pickle.dump(Knn_labels,open('Knn_154','wb'))
    Knn_labels, nearest = KNN(x.PCA[:,:50],digits.train_Labels,test_Images_Center@np.transpose(x.V[:50,:]),10)
    pickle.dump(Knn_labels,open('Knn_50','wb'))
    pickle.dump(nearest,open('Knn_50_nearest','wb'))


def inboth_index(list1,list2):
    # returns a list of index's in list2 but not in list1
    index = np.zeros(list2.shape)
    for i in range(list2.shape[0]):
        if list2[i] not in list1:
            index[i] = 1
    index = np.nonzero(index.astype(int))[0]
    return index