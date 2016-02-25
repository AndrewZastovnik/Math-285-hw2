import numpy as np
import pickle
import pylab as plt


# make a class variable that has the information that i need
class center_matrix:
    # A class to store our information about our centered matrix
    def __init__(self,a,dim=0):
        self.size = a.shape
        # Reshaped as 1,n instead of ,n
        self.centers = np.mean(a,axis=dim).reshape(1,self.size[1])
        self.a_centered = np.subtract(a,np.repeat(self.centers,self.size[dim],dim))
        self.U, self.s, self.V = np.linalg.svd(self.a_centered,full_matrices = True)

def main():
    train_Images = pickle.load(open('mnistTrainI.p', 'rb'))
    #train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
    #test_Images = pickle.load(open('mnistTestI.p', 'rb'))
    #test_labels = pickle.load(open('mnistTestL.p', 'rb'))
    train = center_matrix(train_Images)
    #plt.imshow(train.centers.reshape(28,28))
    #plt.show()
    #plt.imshow(train.a_centered[1,:].reshape(28,28))
    #plt.show()
    pickle.dump(train,open('Training SVD Data'),'wb')

if __name__ == "__main__":
    main()