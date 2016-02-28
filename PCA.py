import numpy as np
import pickle
import pylab as plt
from NearestNeighbors import mfoldX,KNN

class center_matrix_SVD:
    # A class to store our information about our centered matrix
    # center_matrix has 7 atributes
    # .size stores the shape of the original matrix
    # .centers stores the center of the dataset
    # a_centered is the centered original matrix
    # .U .s .V are the SVD decomposition of the centered matrix
    def __init__(self,a,dim=0):
        self.size = a.shape
        # Reshaped as 1,n instead of ,n because that was causing problems
        self.centers = np.mean(a,axis=dim).reshape(1,self.size[1])
        self.a_centered = np.subtract(a,np.repeat(self.centers,self.size[dim],dim))
        self.U, self.s, self.V = np.linalg.svd(self.a_centered,full_matrices = False)
        self.PCA = self.U@np.diagflat(self.s)


def dump_the_svd():
    # Gets SVD for the training dataset
    train_Images = pickle.load(open('mnistTrainI.p', 'rb'))
    train = center_matrix_SVD(train_Images)
    pickle.dump(train,open('Training SVD Data','wb'))

def class_error_rate(pred_labels,true_labels):
    # for calculating the error rate
    # Also returns a index vector with the position of incorrectly labeled images
    error = np.zeros(pred_labels.shape[0])
    error_index = np.zeros((pred_labels.shape[0],pred_labels.shape[1]))
    for i in range(pred_labels.shape[0]):
        error[i] = sum(pred_labels[i] != true_labels)/pred_labels.shape[1]
        print([pred_labels[i] != true_labels])
        plt.scatter(np.arange(60000),[pred_labels[i] != true_labels].shape)
        plt.show()
        error_index[i] = np.arange(pred_labels.shape[1])[pred_labels[i] != true_labels]
    return error, error_index

def MFold_plots():
    # Mfold plots
    err = pickle.load(open('MFolderrors50','rb'))
    err2 = pickle.load(open('MFolderrors','rb'))
    err3 = pickle.load(open('MFolderrors154','rb'))
    plt.figure()
    ax = plt.subplot(111)
    ax.scatter(np.arange(10)+1,err,color = 'Green',s=200,label = '82.5%')
    ax.scatter(np.arange(10)+1,err3,color = 'Yellow',s=150,label = '95%')
    ax.scatter(np.arange(10)+1,err2,color = 'Red',s=100,label = 'Full')
    ax.grid(1)
    plt.title('Scatter plot of the three error rates')
    plt.legend(loc='upper right')
    plt.show()

def do_KNN(x,train_Labels):
    test_Images = pickle.load(open('mnistTestI.p', 'rb'))
    test_Images_Center = np.mean(test_Images,axis=0).reshape(1,test_Images.size[1])
    Knn_labels = KNN(x.PCA,train_Labels,test_Images_Center@x.V,10)
    pickle.dump(Knn_labels,open('Knn_Full','wb'))
    Knn_labels = KNN(x.PCA[:,:154],train_Labels,test_Images_Center@x.V[:,:154],10)
    pickle.dump(Knn_labels,open('Knn_154','wb'))
    Knn_labels = KNN(x.PCA[:,:50],train_Labels,test_Images_Center@x.V[:,:50],10)
    pickle.dump(Knn_labels,open('Knn_50','wb'))

def KNN_Plots():
    # KNN plots
    labels_50 = pickle.load(open('KNN_50','rb'))
    labels_154 = pickle.load(open('KNN_154','rb'))
    labels_Full = pickle.load(open('KNN_Full','rb'))
    test_labels = pickle.load(open('mnistTestL.p', 'rb'))
    error_50, error_50_index = class_error_rate(labels_50,test_labels)
    print(error_50_index)


def main():
    # the if 0 or 1 stuff is so that i can run only the code i want without deleting or commenting out code
    if 0:
        dump_the_svd()
    if 1: # do stuff after getting the svd iformation
        x = pickle.load(open('Training SVD Data','rb'))
        train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
        if 0: #Do m-folds
            merror = mfoldX(x.PCA[:,:50],train_Labels,6,10)
            pickle.dump(merror,open('MFoldErrors154','wb'))
        if 1: #Do K-nearest Neighbors
            do_KNN(x,train_Labels)
    if 0: # Plot stuff
        KNN_Plots()
        MFold_plots()
        plt.imshow(x.PCA[0,:].reshape(28,28),interpolation = 'none')
        plt.show()
        #plt.imshow(train.a_centered[1,:].reshape(28,28))
        #plt.show()
        plt.scatter(np.arange(784),x.PCA[0,:])
        plt.show()
        plt.scatter(np.arange(154),x.U[0,0:154])
        plt.show()
        plt.scatter(np.arange(154),x.s[0:154])
        plt.show()
        plt.imshow(x.V[0,:].reshape(28,28))
        plt.show()


if __name__ == "__main__":
    main()


"""
Plotting code
        train_Images = x.U[:,0:154]@np.diagflat(x.s)[0:154,0:154]
"""