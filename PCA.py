import numpy as np
import pickle
import pylab as plt
from NearestNeighbors import mfoldX

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

def main():
    # skip is a bool list that says what functions I want to run
    skip = [0,1,0]
    if skip[0]:
        dump_the_svd()
    if skip[1]:
        x = pickle.load(open('Training SVD Data','rb'))
        train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
        merror = mfoldX(x.PCA,train_Labels,6,10)
        pickle.dump(merror,open('MFoldErrors','wb'))
    #test_Images = pickle.load(open('mnistTestI.p', 'rb'))
    #test_labels = pickle.load(open('mnistTestL.p', 'rb'))
    if skip[2]:
        plt.imshow(x.centers.reshape(28,28))
        plt.show()
        #plt.imshow(train.a_centered[1,:].reshape(28,28))
        #plt.show()


if __name__ == "__main__":
    main()


"""
Plotting code
        train_Images = x.U[:,0:154]@np.diagflat(x.s)[0:154,0:154]
        plt.scatter(np.arange(154),train_Images[0,:])
        plt.show()
        plt.scatter(np.arange(154),x.U[0,0:154])
        plt.show()
        plt.scatter(np.arange(154),x.s[0:154])
        plt.show()
        plt.imshow(x.V[0,:].reshape(28,28))
        plt.show()
"""