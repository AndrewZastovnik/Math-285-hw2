from PCA import center_matrix_SVD,mnist  # Imports some functions from our PCA file
import numpy as np  # Numpy very important
from NearestNeighbors import mfoldX  # Get our NN functions
import pickle  # store output and get input
import pylab as plt

def main(): # Our main function
    digits = mnist() # Creates a class with our mnist images and labels
    if open('Training SVD Data','rb')._checkReadable() == 0: # Check if file exist create it if it doesn't
        print("im here")
        x = center_matrix_SVD(digits.train_Images) # Creates a class with our svd and associated info
        pickle.dump(x,open('Training SVD Data','wb'))
    else:
        x = pickle.load(open('Training SVD Data','rb'))
    if 0: # change 0 to 1 if you want to run this agian
        merror = mfoldX(x.PCA[:,:],digits.train_Labels,6,10) # Run X-validation and return error rates for the full dataset
        pickle.dump(merror,open('MFoldErrors','wb')) # Put our error rates in a file
        merror = mfoldX(x.PCA[:,:154],digits.train_Labels,6,10) # For the 95% dataset
        pickle.dump(merror,open('MFoldErrors154','wb'))
        merror = mfoldX(x.PCA[:,:50],digits.train_Labels,6,10) # for the 82.5% dataset
        pickle.dump(merror,open('MFoldErrors50','wb'))
    MFold_plots(x) # Makes graphs from our data

def MFold_plots(x):
    # A function for plotting output from our m fold data
    err4_var = round(np.sum(np.power(x.s[:25],2))/np.sum(np.power(x.s,2))*100,1) # Calulate the reduced variance
    err4_lab = "%s%%" % str(err4_var)
    err5_var = round(np.sum(np.power(x.s[:40],2))/np.sum(np.power(x.s,2))*100,1)
    err5_lab = "%s%%" % str(err5_var)
    err6_var = round(np.sum(np.power(x.s[:45],2))/np.sum(np.power(x.s,2))*100,1)
    err6_lab = "%s%%" % str(err6_var)
    err = pickle.load(open('MFolderrors50', 'rb'))  # Loads the data we saved eariler
    err2 = pickle.load(open('MFolderrors', 'rb'))  # I Know this numbering doesn't really make sense
    err3 = pickle.load(open('MFolderrors154', 'rb'))
    err4 = pickle.load(open('MFolderrors25', 'rb'))
    err5 = pickle.load(open('MFolderrors40', 'rb'))
    err6 = pickle.load(open('MFolderrors45', 'rb'))
    plt.figure()
    plt.plot(np.arange(10)+1, err, color='Green', marker='o', markersize=10, label='82.5%')  #plots the 82.5%
    plt.plot(np.arange(10)+1, err3, color='Yellow', marker='o', markersize=10, label='95%')
    plt.plot(np.arange(10)+1, err2, color='Red', marker='o', markersize=10, label='Full')
    plt.plot(np.arange(10)+1, err4, color='Blue', marker='o', markersize=10, label=err4_lab)
    plt.plot(np.arange(10)+1, err5, color='Purple', marker='o', markersize=10, label=err5_lab)
    plt.plot(np.arange(10)+1, err6, color='Black', marker='o', markersize=10, label=err6_lab)
    plt.grid(1) # Turns the grid on
    plt.title('Scatter plot of error rates')
    plt.legend(loc='upper right') # Puts a legend on the plot
    plt.show()

if __name__ == "__main__":
    main()