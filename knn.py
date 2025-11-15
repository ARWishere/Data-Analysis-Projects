'''knn.py
K-Nearest Neighbors algorithm for classification
Andrew Welling
CS 251: Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classifier import Classifier

class KNN(Classifier):
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor
        '''
        super().__init__(num_classes)

        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.
        '''
        pred = []
        for sample in data:
            distances = np.linalg.norm(self.exemplars - sample,axis=1)
            closest_inds = np.argsort(distances)[:k] # use k best samples
            labels = self.classes[closest_inds]
            counts = np.bincount(labels.astype(int)) # count classes
            pred_class = np.argmax(counts) # create prediction based on highest count
            pred.append(pred_class)

        return np.array(pred)



    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.
        '''
        colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
        cmp = ListedColormap(colors)
        samples_1 = np.linspace(-40, 40,num=n_sample_pts)
        samples_2 = np.linspace(-40, 40, num=n_sample_pts)
        x,y = np.meshgrid(samples_1, samples_2)
        x_flat = x.flatten()
        y_flat = y.flatten()
        x_reshape = x_flat.reshape((x_flat.shape[0], 1))
        y_reshape = y_flat.reshape((x_flat.shape[0], 1))
        xy_samples = np.hstack((x_reshape, y_reshape))
        y_pred = self.predict(xy_samples,k)
        y_pred = y_pred.reshape(n_sample_pts,n_sample_pts)
        mesh = plt.pcolormesh(x,y,y_pred,cmap=cmp)
        plt.colorbar(mesh, label='prediction colorbar')




