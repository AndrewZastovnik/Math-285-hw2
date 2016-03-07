from PCA import center_matrix_SVD,mnist,class_error_rate  # Imports some functions from our PCA file
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from NearestNeighbors import local_kmeans_class  # Get our NN functions
from TicToc import tic,toc
import pylab as plt


def main():
    digits = mnist() # Creates a class with our mnist images and labels
    if open('Training SVD Data','rb')._checkReadable() == 0: # Check if file exist create it if it doesn't
        x = center_matrix_SVD(digits.train_Images) # Creates a class with our svd and associated info
        pickle.dump(x,open('Training SVD Data','wb'))
    else:
        x = pickle.load(open('Training SVD Data','rb'))  # If we already have the file just load it
    if 1: # if this is zero skip
        test_Images_Center = np.subtract(digits.test_Images,np.repeat(x.centers,digits.test_Images.shape[0],0))
        tic()
        myLDA = LDA()  # Create a new instance of the LDA class
        new_train = myLDA.fit_transform(x.PCA[:,:154],digits.train_Labels)  # It will fit based on x.PCA
        new_test = myLDA.transform(test_Images_Center@np.transpose(x.V[:154,:])) # get my transformed test dataset
        Knn_labels = local_kmeans_class(new_train,digits.train_Labels,new_test,10) # Run kNN on the new data
        toc()
        pickle.dump(Knn_labels,open('Loc_kmeans_fda_lab','wb'))

    fda = pickle.load(open('Loc_kmeans_fda_lab','rb'))
    labels_Full = pickle.load(open('KNN_Full','rb'))
    loc_full = pickle.load(open('Loc_kmeans_Full_lab','rb'))
    errors_fda,ind_fda = class_error_rate(np.transpose(fda),digits.test_labels)
    errors_near,ind_near = class_error_rate(labels_Full,digits.test_labels)
    errors_full,ind_full = class_error_rate(np.transpose(loc_full),digits.test_labels)
    labels_50 = pickle.load(open('KNN_50','rb'))
    errors_50,ind_50 = class_error_rate(labels_50,digits.test_labels)
    print(errors_full)
    plt.figure()
    plt.plot(np.arange(10)+1, errors_fda, color='Green', marker='o', markersize=10, label='fda Kmeans')  #plots the 82.5%
    plt.plot(np.arange(10)+1, errors_near, color='Blue', marker='o', markersize=10, label='kNN')
    plt.plot(np.arange(10)+1, errors_full, color='Yellow', marker='o', markersize=10, label='Full Kmeans')
    plt.plot(np.arange(10)+1, errors_50, color='Red', marker='o', markersize=10, label='kNN 50')
    axes = plt.gca()
    axes.set_ylim([0.015,0.12])
    plt.grid(1) # Turns the grid on
    plt.title('Plot of Local Kmeans with FDA Error rates')
    plt.legend(loc='upper right')  # Puts a legend on the plot
    plt.show()
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
if __name__ == "__main__":
    main()