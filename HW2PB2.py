from PCA import center_matrix_SVD,mnist,class_error_rate,inboth_index  # Imports some functions from our PCA file
import pylab as plt
from NearestNeighbors import KNN  # Get our NN functions
import numpy as np
import pickle

def main():
    digits = mnist() # Creates a class with our mnist images and labels
    if open('Training SVD Data','rb')._checkReadable() == 0: # Check if file exist create it if it doesn't
        print("im here")
        x = center_matrix_SVD(digits.train_Images) # Creates a class with our svd and associated info
        pickle.dump(x,open('Training SVD Data','wb'))
    else:
        x = pickle.load(open('Training SVD Data','rb'))
    if 0:  # change to 1 if you want to rerun the knn stuff
        do_KNN(x,digits)
    KNN_Plots(x,digits)

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

def KNN_Plots(x,digits):
    # KNN plots
    labels_50 = pickle.load(open('KNN_50','rb'))
    labels_154 = pickle.load(open('KNN_154','rb'))
    labels_Full = pickle.load(open('KNN_Full','rb'))
    nearest_50 = pickle.load(open('Knn_50_nearest','rb'))
    nearest_154 = pickle.load(open('Knn_154_nearest','rb'))
    nearest_Full = pickle.load(open('Knn_Full_nearest','rb'))
    error_50, error_50_index = class_error_rate(labels_50,digits.test_labels)
    error_154, error_154_index = class_error_rate(labels_154,digits.test_labels)
    error_Full, error_Full_index = class_error_rate(labels_Full,digits.test_labels)
    print(error_50)
    print(error_154)
    print(error_Full)
    plt.figure()
    plt.bar([0,1,2],[error_50[2],error_154[2],error_Full[2]])
    plt.grid(1)
    plt.title('Bar Plot of Error Rates')
    plt.legend(loc='upper right')
    plt.show()
    error_50_index = np.asarray(np.where(error_50_index[2]))
    error_154_index = np.asarray(np.where(error_154_index.astype(int)[2]))
    error_Full_index = np.asarray(np.where(error_Full_index.astype(int)[2]))
    error_in_50_Full = error_50_index[0,inboth_index(error_Full_index[0],error_50_index[0])]
    # This is a loop that looks through digits the 50 dim PCA got correct but the full didn't
    for i in range(error_in_50_Full.shape[0]):
        j = error_in_50_Full[i]
        test_Images_Center = np.subtract(digits.test_Images,np.repeat(x.centers,digits.test_Images.shape[0],0))
        y = test_Images_Center@np.transpose(x.V[:50,:])
        weighted_y = y[:,:50]@x.V[:50,:] + x.centers
        plt.subplot(2, 3, 1)
        plt.imshow(weighted_y[j].reshape(28,28),cmap='gray',interpolation = 'none')
        plt.axis('off')
        plt.title("In 50 %d Truth %d " % (np.asscalar(labels_50[2,j]), np.asscalar(digits.test_labels[j])))
        y = test_Images_Center@np.transpose(x.V[:154,:])
        weighted_y2 = y[:,:154]@x.V[:154,:] + x.centers
        plt.subplot(2, 3, 2)
        plt.imshow(weighted_y2[j].reshape(28,28),cmap='gray',interpolation = 'none')
        plt.axis('off')
        plt.title("in 150 %d Truth %d " % (np.asscalar(labels_154[2,j]), np.asscalar(digits.test_labels[j])))
        plt.subplot(2, 3, 3)
        plt.imshow(digits.test_Images[j].reshape(28,28),cmap='gray')
        plt.axis('off')
        plt.title("in Full %d Truth %d " % (np.asscalar(labels_Full[2,j]), np.asscalar(digits.test_labels[j])))
        plt.subplot(2, 3, 4)
        weighted_x = x.PCA[:,:50]@x.V[:50,:] + x.centers
        myimage = np.hstack((weighted_x[nearest_50[j,0]].reshape(28,28),
                             weighted_x[nearest_50[j,1]].reshape(28,28),weighted_x[nearest_50[j,2]].reshape(28,28)))
        plt.imshow(myimage,cmap='gray')
        plt.title(np.array_str(digits.train_Labels[nearest_50[j,:3].astype(int)]))
        plt.axis('off')
        plt.subplot(2, 3, 5)
        weighted_x = x.PCA[:,:154]@x.V[:154,:] + x.centers
        myimage = np.hstack((weighted_x[nearest_154[j,0]].reshape(28,28),
                             weighted_x[nearest_154[j,1]].reshape(28,28),weighted_x[nearest_154[j,2]].reshape(28,28)))
        plt.imshow(myimage,cmap='gray')
        plt.title(np.array_str(digits.train_Labels[nearest_154[j,:3].astype(int)]))
        plt.axis('off')
        plt.subplot(2, 3, 6)
        weighted_x = x.a_centered + x.centers
        myimage = np.hstack((weighted_x[nearest_Full[j,0]].reshape(28,28),
                             weighted_x[nearest_Full[j,1]].reshape(28,28),weighted_x[nearest_Full[j,2]].reshape(28,28)))
        plt.imshow(myimage,cmap='gray')
        plt.title(np.array_str(digits.train_Labels[nearest_Full[j,:3].astype(int)]))
        print(np.array_str(nearest_Full[j,:3].astype(int)))
        print(np.array_str(nearest_154[j,:3].astype(int)))
        print(np.array_str(nearest_50[j,:3].astype(int)))
        plt.axis('off')
        plt.show()



if __name__ == "__main__":
    main()