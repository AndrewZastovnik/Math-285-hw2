import numpy as np
import pickle
import pylab as plt
from NearestNeighbors import mfoldX,KNN,local_kmeans_class
from TicToc import tic,toc

class center_matrix_SVD:
    # A class to store our information about our centered matrix
    # center_matrix has 7 atributes
    # .size stores the shape of the original matrix
    # .centers stores the center of the dataset
    # a_centered is the centered original matrix
    # .U .s .V are the SVD decomposition of the centered matrix
    def __init__(self,a,dim=0):
        self.size = a.shape # Gets and stores the shape of a not sure it is really necessary
        self.centers = np.mean(a,axis=dim).reshape(1,self.size[1])# Reshaped as 1,n instead of ,n because that was causing problems
        self.a_centered = np.subtract(a,np.repeat(self.centers,self.size[dim],dim)) #Creates an atribute a_centered to store the  centered a matrix
        self.U, self.s, self.V = np.linalg.svd(self.a_centered,full_matrices = False) # Runs SVD on our centered a matrix
        self.PCA = self.U@np.diagflat(self.s) # stores the U*S matrix in atribute PCA since s is diagnol we can just take rows out of this instead of recalulating PCA for reduced dims


def dump_the_svd():
    # Gets SVD for the training dataset
    train_Images = pickle.load(open('mnistTrainI.p', 'rb'))
    train = center_matrix_SVD(train_Images)
    pickle.dump(train,open('Training SVD Data','wb')) # You won't be able to load this unless you have the center_matrix_SVD class avalible

def class_error_rate(pred_labels,true_labels):
    # for calculating the error rate
    # Also returns a index vector with the position of incorrectly labeled images
    error = np.zeros(pred_labels.shape[0])
    error_index = np.zeros((pred_labels.shape[0],pred_labels.shape[1]))
    for i in range(pred_labels.shape[0]):
        error[i] = sum(pred_labels[i] != true_labels)/pred_labels.shape[1]
        error_index[i] = 1 - np.isclose(pred_labels[i],true_labels)
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
    #code to run knn for the three values of s
    test_Images = pickle.load(open('mnistTestI.p', 'rb'))
    test_Images_Center = np.subtract(test_Images,np.repeat(x.centers,test_Images.shape[0],0))
    #Knn_labels = KNN(x.PCA,train_Labels,test_Images_Center@np.transpose(x.V[:,:]),10)
    #pickle.dump(Knn_labels,open('Knn_Full','wb'))
    #Knn_labels = KNN(x.PCA[:,:154],train_Labels,test_Images_Center@np.transpose(x.V[:154,:]),10)
    #pickle.dump(Knn_labels,open('Knn_154','wb'))
    Knn_labels, nearest = KNN(x.PCA[:,:50],train_Labels,test_Images_Center@np.transpose(x.V[:50,:]),10)
    pickle.dump(Knn_labels,open('Knn_50','wb'))
    pickle.dump(nearest,open('Knn_50_nearest','wb'))

def KNN_Plots(x):
    # KNN plots
    labels_50 = pickle.load(open('KNN_50','rb'))
    labels_154 = pickle.load(open('KNN_154','rb'))
    labels_Full = pickle.load(open('KNN_Full','rb'))
    test_labels = pickle.load(open('mnistTestL.p', 'rb'))
    nearest_50 = pickle.load(open('Knn_50_nearest','rb'))
    print(nearest_50.shape)
    error_50, error_50_index = class_error_rate(labels_50,test_labels)
    error_154, error_154_index = class_error_rate(labels_154,test_labels)
    error_Full, error_Full_index = class_error_rate(labels_Full,test_labels)
    print(test_labels[33])
    print(labels_50[2,33])
    error_50_index = np.asarray(np.where(error_50_index[2]))
    error_154_index = np.asarray(np.where(error_154_index.astype(int)[2]))
    error_Full_index = np.asarray(np.where(error_Full_index.astype(int)[2]))
    error_in_50_Full = error_Full_index[0,inboth_index(error_50_index[0],error_Full_index[0])]
    test_Images = pickle.load(open('mnistTestI.p', 'rb'))
    for i in range(error_in_50_Full.shape[0]):
        j = error_in_50_Full[i]
        test_Images_Center = np.subtract(test_Images,np.repeat(x.centers,test_Images.shape[0],0))
        y = test_Images_Center@np.transpose(x.V[:50,:])
        weighted_y = y[:,:50]@x.V[:50,:] + x.centers
        plt.subplot(2, 3, 1)
        plt.imshow(weighted_y[j].reshape(28,28),interpolation = 'none')
        plt.axis('off')
        plt.title("In 50 %d Truth %d " % (np.asscalar(labels_50[2,j]), np.asscalar(test_labels[j])))
        y = test_Images_Center@np.transpose(x.V[:154,:])
        weighted_y2 = y[:,:154]@x.V[:154,:] + x.centers
        plt.subplot(2, 3, 2)
        plt.imshow(weighted_y2[j].reshape(28,28),cmap='gray',interpolation = 'none')
        plt.axis('off')
        plt.title("in 150 %d Truth %d " % (np.asscalar(labels_154[2,j]), np.asscalar(test_labels[j])))
        plt.subplot(2, 3, 3)
        plt.imshow(test_Images[j].reshape(28,28),cmap='gray')
        plt.axis('off')
        plt.title("in Full %d Truth %d " % (np.asscalar(labels_Full[2,j]), np.asscalar(test_labels[j])))
        #plt.subplot(2, 3, 2)
        #train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
        #plt.imshow(x.a_centered[j].reshape(28,28),cmap='gray')
        plt.show()

def inboth_index(list1,list2):
    # returns a list of index's in list2 but not in list1
    index = np.zeros(list2.shape)
    for i in range(list2.shape[0]):
        if list2[i] not in list1:
            index[i] = 1
    index = np.nonzero(index.astype(int))[0]
    return index

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
        if 0: #Do K-nearest Neighbors
            do_KNN(x,train_Labels)
        if 1:
            test_Images = pickle.load(open('mnistTestI.p', 'rb'))
            test_Images_Center = np.subtract(test_Images,np.repeat(x.centers,test_Images.shape[0],0))
            tic()
            labels, nearest = local_kmeans_class(x.PCA[:,:50],train_Labels,test_Images_Center@np.transpose(x.V[:50,:]),10)
            toc()
            pickle.dump(labels,open('Loc_kmeans_lab','wb'))
            pickle.dump(nearest,open('Loc_kmeans_near','wb'))
    if 0: # Plot stuff
        x = pickle.load(open('Training SVD Data','rb'))
        KNN_Plots(x)
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
                plt.imshow((x.a_centered[0]@np.transpose(x.V[:,:])).reshape(28,28))
        plt.show()
        plt.imshow(x.PCA[0,:].reshape(28,28))
        plt.show()
"""