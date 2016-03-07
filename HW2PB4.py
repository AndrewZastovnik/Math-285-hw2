from PCA import center_matrix_SVD,mnist,class_error_rate  # Imports some functions from our PCA file
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from NearestNeighbors import KNN  # Get our NN functions
from TicToc import tic,toc
import pylab as plt


def main():
    digits = mnist() # Creates a class with our mnist images and labels
    if open('Training SVD Data','rb')._checkReadable() == 0: # Check if file exist create it if it doesn't
        print("im here")   # Just wanted to check if it was going in here
        x = center_matrix_SVD(digits.train_Images) # Creates a class with our svd and associated info
        pickle.dump(x,open('Training SVD Data','wb'))
    else:
        x = pickle.load(open('Training SVD Data','rb'))  # If we already have the file just load it
    if 0: # if this is zero skip
        test_Images_Center = np.subtract(digits.test_Images,np.repeat(x.centers,digits.test_Images.shape[0],0))
        tic()
        myLDA = LDA()  # Create a new instance of the LDA class
        new_train = myLDA.fit_transform(x.PCA[:,:154],digits.train_Labels)  # It will fit based on x.PCA
        new_test = myLDA.transform(test_Images_Center@np.transpose(x.V[:154,:])) # get my transformed test dataset
        Knn_labels, nearest = KNN(new_train,digits.train_Labels,new_test,10) # Run kNN on the new data
        toc()
        pickle.dump(Knn_labels,open('FDAKNN_Lables','wb'))
        pickle.dump(nearest,open('FDAKNN_neastest','wb'))
    fda = pickle.load(open('FDAKNN_Lables','rb'))
    labels_Full = pickle.load(open('KNN_Full','rb'))
    labels_50 = pickle.load(open('KNN_50','rb'))
    errors_fda,ind_fda = class_error_rate(fda,digits.test_labels)
    errors_near,ind_near = class_error_rate(labels_Full,digits.test_labels)
    errors_50,ind_50 = class_error_rate(labels_50,digits.test_labels)
    plt.figure()
    plt.plot(np.arange(10)+1, errors_fda, color='Green', marker='o', markersize=10, label='fda')  #plots the 82.5%
    plt.plot(np.arange(10)+1, errors_near, color='Blue', marker='o', markersize=10, label='kNN')
    plt.plot(np.arange(10)+1, errors_50, color='Yellow', marker='o', markersize=10, label='kNN 50')
    plt.grid(1) # Turns the grid on
    plt.title('Plot of Knn with FDA Error rates')
    plt.legend(loc='upper right')  # Puts a legend on the plot
    plt.show()
    print(confusion_matrix(digits.test_labels,labels_Full[5]))
    print(confusion_matrix(digits.test_labels,fda[5]))
    print(confusion_matrix(digits.test_labels,labels_50[5]))
    """
    project_back(x,digits)


def project_back(x,digits):
    myLDA = LDA()
    new_train = myLDA.fit_transform(x.PCA[:,:154],digits.train_Labels)
    print(new_train.shape)
    m = 0
    n = 1
    plt.figure()
    plt.scatter(new_train[digits.train_Labels == 0,m],new_train[digits.train_Labels == 0,n], color='Green', s= 1)
    plt.scatter(new_train[digits.train_Labels == 1,m],new_train[digits.train_Labels == 1,n], color='Blue', s= 1)
    plt.scatter(new_train[digits.train_Labels == 2,m],new_train[digits.train_Labels == 2,n], color='Red', s= 1)
    plt.scatter(new_train[digits.train_Labels == 3,m],new_train[digits.train_Labels == 3,n], color='Purple', s= 1)
    plt.scatter(new_train[digits.train_Labels == 4,m],new_train[digits.train_Labels == 4,n], color='Black', s= 1)
    plt.scatter(new_train[digits.train_Labels == 5,m],new_train[digits.train_Labels == 5,n], color='Brown', s= 1)
    plt.scatter(new_train[digits.train_Labels == 6,m],new_train[digits.train_Labels == 6,n], color='Silver', s= 1)
    plt.scatter(new_train[digits.train_Labels == 7,m],new_train[digits.train_Labels == 7,n], color='Cyan', s= 1)
    plt.show()
    y = new_train@myLDA.coef_[:9,:] # I really don't know if this will work since there are 10 coef things
    weighted_y2 = y[:,:154]@x.V[:154,:] + x.centers
    plt.imshow(weighted_y2[0,:].reshape(28,28))
    plt.show()
    """
if __name__ == "__main__":
    main()