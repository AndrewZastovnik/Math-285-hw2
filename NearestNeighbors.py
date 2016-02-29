import numpy as np
from TicToc import tic
from TicToc import toc

def KNN(I, L, x, k,weights = 1):
    """
    I is the matrix of obs
    L are the labels
    x is what we are trying to classify
    k are how many neighbors to look at or whatever
    first we want to create a matrix of distances from each object we want to classify to every object in our training set
    """
    from scipy import stats
    from scipy.spatial.distance import cdist
    sizex = len(np.atleast_2d(x))
    label = np.zeros((k,sizex))
    nearest = 0
    for rowsx in range(0, sizex):
        dists = cdist(I, np.atleast_2d(x[rowsx]), metric='euclidean')
        # Now we should have all our distances in our dist array
        # Next find the k closest neighbors of x
        k_smallest = np.argpartition(dists,tuple(range(1,k+1)),axis=None)
        nearest = k_smallest[:k+1]
        # The next step is to use this info to classify each unknown obj
        # if we don't want to use weights weights should equal 1
        if weights == 1:
            for i in range(0,k):
                label[i,rowsx] = stats.mode(L[k_smallest[:i+1]])[0]
        else:
            labs = np.unique(L)
            for i in range(k):
                lab_weighted = np.zeros(np.unique(L).shape[0])
                d = dists[k_smallest[:i+2]][:,0]
                weights = weight_function(d)
                for p in range(0,labs.shape[0]):
                    indices = inboth(np.arange(0,L.shape[0])[L == labs[p]],k_smallest[:i+2])
                    lab_weighted[p]= np.sum(np.multiply(weights,indices))
                label[i,rowsx] = labs[np.argmax(lab_weighted)]
        if rowsx % 1000 == 1:
            print(rowsx)
    return label, nearest

def weight_function(d):
    #takes a distance vector d and computes the associated linear weights
    weights = np.add(np.divide(d, np.subtract(np.min(d),np.max(d))),1-np.min(d)/np.subtract(np.min(d),np.max(d)))
    return weights

def inboth(list1,list2):
    # returns a list of 1's and 0's the same length as list2 where 1's mean that index is also in list1
    index = np.zeros(list2.shape)
    for i in range(list2.shape[0]):
        if list2[i] in list1:
            index[i] = 1
    return index

def class_error_rate(pred_labels,true_labels):
    # for calculating the error rate
    error = np.zeros(pred_labels.shape[0])
    for i in range(pred_labels.shape[0]):
        error[i] = sum(pred_labels[i] != true_labels)/pred_labels.shape[1]
    return error

def mfoldX(I, L, m, maxk):
    # I is the trainset
    # L is the Training Labels
    # m is the number of folds
    # maxk is the largest value of k we wish to test
    # first thing to acomplish is to randomly divide the data into m parts
    indices = np.random.permutation(I.shape[0])
    jump = round(len(L) / m)
    I_index = indices[:jump]
    L_index = indices[:jump]
    for n in range(1, m - 1):
        I_index = np.dstack((I_index, indices[n * jump:(n + 1) * jump]))
        L_index= np.dstack((L_index, indices[n * jump:(n + 1) * jump]))
    I_index = np.dstack((I_index, indices[(m-1) * jump:]))
    print(I_index.shape)
    L_index = np.dstack((L_index, indices[(m-1) * jump:]))
    # now data should be all nice and divided up we need to do something else
    error = np.zeros(maxk)
    for n in range(0, m):
        mask = np.ones(m,dtype=bool)
        mask[n]=0
        notn = np.arange(0,m)[mask]
        Ipt = I[I_index[:,:,notn].reshape(((m-1)*I_index.shape[1]))]
        Lpt = L[I_index[:,:,notn].reshape(((m-1)*I_index.shape[1]))]
        label = KNN(Ipt,Lpt ,I[I_index[:,:,n].reshape(I_index.shape[1])],10)
        for k in range(10):
            error[k] = error[k] + sum((label[k] != L[L_index[:,:,n]])[0])
    error = error / (len(L))
    return error

def local_kmeans_class(I, L, x, k):
    from scipy.spatial.distance import cdist

    sizex = len(np.atleast_2d(x))
    label = np.zeros((sizex,k))
    for rowsx in range(0, sizex):
        tic()
        dists = cdist(I, np.atleast_2d(x[rowsx]), metric='euclidean')
        toc()
        center = np.zeros((10,k,28*28))
        label_order = np.unique(L)
        l=0
        tic()
        thing = np.zeros((k,28*28))
        for labs in np.unique(L):
            indices = L == labs
            k_smallest = np.argpartition(dists[indices],tuple(range(1,k)),axis=None)
            for i in range(0,k):
                M = I[indices]
                #center[l,i,:] = np.average(M[k_smallest[:i+1]],axis = 0)
                if i == 0:
                    thing[i] = M[k_smallest[i+1]]
                else:
                    thing[i] = thing[i-1] + M[k_smallest[i+1]]
            center[l,:,:] = np.divide(thing,np.repeat(np.arange(1,11).reshape(10,1),28*28,axis=1))
            l+=1
        toc()
        for i in range(k):
            #print(cdist(center[:,i,:], np.atleast_2d(x[rowsx]), metric='euclidean'))
            dists2center = cdist(center[:,i,:], np.atleast_2d(x[rowsx]), metric='euclidean')
            k_smallest = np.argpartition(dists2center,tuple(range(1)),axis=None)
            label[rowsx,i] = label_order[k_smallest[0]]
    return label

"""
import pickle
import pandas
import pylab as plt
def main():
    train_Images = pickle.load(open('mnistTrainI.p', 'rb'))
    train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
    test_Images = pickle.load(open('mnistTestI.p', 'rb'))
    test_labels = pickle.load(open('mnistTestL.p', 'rb'))
    skip = 1
    if skip == 0:
        label = KNN(train_Images, train_Labels, test_Images[:10], 12,'yesplease')
        pickle.dump(label, open('kNNWeight.p', 'wb'))
    else:
        label = pickle.load(open('kNNWeight.p', 'rb'))
    errors = class_error_rate(label,test_labels)
    plt.plot(range(12),errors)
    plt.show()

if __name__ == "__main__":
    main()
"""