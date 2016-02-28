import numpy as np
import pickle
from TicToc import tic
from TicToc import toc


def KNN(I, L, x, k,weights = 1):
    from scipy import stats
    from scipy.spatial.distance import cdist
    """
    I is the matrix of obs
    L are the labels
    x is what we are trying to classify
    k are how many neighbors to look at or whatever
    first we want to create a matrix of distances from each
    object we want to classify to every object in our training set
    """
    sizex = len(np.atleast_2d(x))
    label = np.zeros((k,sizex))
    for rowsx in range(0, sizex):
        tic()
        dists = cdist(I, np.atleast_2d(x[rowsx]), metric='euclidean')
        # Now we should have all our distances in our dist array
        # The next step is to use this info to classify each unknown obj
        k_smallest = np.argpartition(dists,tuple(range(1,k+1)),axis=None)
        if weights == 1:
            for i in range(0,k):
                label[i,rowsx] = stats.mode(L[k_smallest[:i+1]])[0]
        else:
            labs = np.unique(L)
            myimage = x[rowsx].reshape(28,28)
            for i in range(k):
                lab_weighted = np.zeros(np.unique(L).shape[0])
                d = dists[k_smallest[:i+2]][:,0]
                weight_function = np.add(np.divide(d, np.subtract(np.min(d),np.max(d))),1-np.min(d)/np.subtract(np.min(d),np.max(d)))
                for p in range(0,labs.shape[0]):
                    indices = inboth(np.arange(0,L.shape[0])[L == labs[p]],k_smallest[:i+2])
                    lab_weighted[p]= np.sum(np.multiply(weight_function,indices))
                label[i,rowsx] = labs[np.argmax(lab_weighted)]
        toc()
        print(rowsx)
    return label

def plot_weights(weight_function,I,myimage,k_smallest,i):
    import pylab as plt
    plt.subplot(2,1,1)
    plt.plot(range(weight_function.shape[0]),weight_function)
    plt.subplot(2,1,2)
    myimage = np.hstack((myimage,I[k_smallest[i]].reshape(28,28)))
    plt.imshow(myimage,cmap = 'gray',)
    plt.show()

def inboth(list1,list2):
    index = np.zeros(list2.shape)
    for i in range(list2.shape[0]):
        if list2[i] in list1:
            index[i] = 1
    return index

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

def class_error_rate(pred_labels,true_labels):
    error = np.zeros(pred_labels.shape[0])
    for i in range(pred_labels.shape[0]):
        error[i] = sum(pred_labels[i] != true_labels)/pred_labels.shape[1]
    return error

train_Images = pickle.load(open('mnistTrainI.p', 'rb'))
train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
test_Images = pickle.load(open('mnistTestI.p', 'rb'))
test_labels = pickle.load(open('mnistTestL.p', 'rb'))
"""
m = local_kmeans_class(train_Images,train_Labels,test_Images[:100],10)
pickle.dump(m, open('localkmeans.p', 'wb'))
label = KNN(train_Images, train_Labels, test_Images[:10], 12,0)
print(label)
#pickle.dump(label, open('KNN_city.p', 'wb'))
"""
label = pickle.load(open('kNNWeight.p', 'rb'))
errors = class_error_rate(label,test_labels)
import pylab as plt
plt.plot(range(12),errors)
plt.show()
"""
label = pickle.load(open('kNNWeight.p', 'rb'))
from sklearn.metrics import confusion_matrix
import pandas
print(test_labels.shape)
x = confusion_matrix(test_labels,label[2])
error = np.zeros(10)
for n in range(10):
    error[n] = 1-x[n,n]/(np.sum(x,axis=0)[n])
print(error)
import matplotlib.pyplot as plt
plt.plot(range(10),error)
plt.show()
print(pandas.DataFrame(x,range(10),range(10)))
"""
"""
tic()
m = mfoldX(train_Images[:6000], train_Labels[:6000], 6, 10)
print(m)
toc()
pickle.dump(m, open('kisfive.p', 'wb'))
"""
"""
import matplotlib.pyplot as plt
m = pickle.load(open('kisfive.p', 'rb'))
plt.plot(range(1,11),m)
plt.show()
"""
"""
from sklearn import neighbors, datasets

n_neighbors = 15
train_Images = pickle.load(open('mnistTrainI.p', 'rb'))
train_Labels = pickle.load(open('mnistTrainL.p', 'rb'))
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(train_Images[10000:],train_Labels[10000:])
Z = clf.predict(train_Images[:10000])
"""
