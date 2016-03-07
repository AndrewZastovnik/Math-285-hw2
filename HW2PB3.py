from PCA import center_matrix_SVD,mnist,class_error_rate  # Imports some functions from our PCA file
import numpy as np
from NearestNeighbors import local_kmeans_class  # Get our local kmeans function
import pickle
from TicToc import tic,toc
import pylab as plt

def main():
    digits = mnist() # Creates a class with our mnist images and labels
    if open('Training SVD Data','rb')._checkReadable() == 0: # Check if file exist create it if it doesn't
        print("im here")
        x = center_matrix_SVD(digits.train_Images) # Creates a class with our svd and associated info
        pickle.dump(x,open('Training SVD Data','wb'))
    else:
        x = pickle.load(open('Training SVD Data','rb'))  # If we already have the file just load it
    if 0:
        test_Images_Center = np.subtract(digits.test_Images,np.repeat(x.centers,digits.test_Images.shape[0],0))
        tic()
        labels = local_kmeans_class(x.PCA[:,:50],digits.train_Labels,test_Images_Center@np.transpose(x.V[:50,:]),10)
        toc()
        pickle.dump(labels,open('Loc_kmeans_50_lab','wb'))
    loc_full = pickle.load(open('Loc_kmeans_Full_lab','rb'))
    loc_50 = pickle.load(open('Loc_kmeans_50_lab','rb'))
    labels_Full = pickle.load(open('KNN_Full','rb'))
    # Have to transpose these because they came out backwards should fix if i use this agian
    errors_full,ind_full = class_error_rate(np.transpose(loc_full),digits.test_labels)
    errors_50,ind_50 = class_error_rate(np.transpose(loc_50),digits.test_labels)
    errors_near,ind_50 = class_error_rate(labels_Full,digits.test_labels)
    plt.figure()
    plt.plot(np.arange(10)+1, errors_full, color='Green', marker='o', markersize=10, label='Full')  #plots the 82.5%
    plt.plot(np.arange(10)+1, errors_50, color='Yellow', marker='o', markersize=10, label='82.5%')
    plt.plot(np.arange(10)+1, errors_near, color='Blue', marker='o', markersize=10, label='kNN')
    plt.grid(1) # Turns the grid on
    plt.title('Plot of local KNN Error rates')
    plt.legend(loc='upper right') # Puts a legend on the plot
    plt.show()

if __name__ == "__main__":
    main()