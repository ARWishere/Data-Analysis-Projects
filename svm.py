'''svm.py
Support vector machines
Andrew Welling
CS 252: Mathematical Data Analysis and Visualization
Spring 2025
'''
import numpy as np

from classifier import Classifier


class SVM(Classifier):
    '''A support vector machine classifier'''

    def __init__(self):
        '''SVM constructor'''
        super().__init__(2)

        # c: ndarray. shape=(M,). "Slope" coefficient for each training sample feature in the optimal separating
        # hyperplane equation.
        self.c = None

        # intercept: float. Intercept of the optimal separating hyperplane.
        self.intercept = None

    def compute_scores(self, features):
        '''Compute the SVM scores for the data `features`.

        Parameters:
        -----------
        features: ndarray. shape=(N, M).
            The data features.

        Returns:
        -----------
        ndarray: shape=(N,).
            The SVM score for each sample.
        '''
        return features @ self.c + self.intercept

    def hinge_loss(self, features, y, reg=0):
        '''Compute the SVM Hinge loss for the data `features`.

        Parameters:
        -----------
        features: ndarray. shape=(N, M).
            The data features.
        y: ndarray. shape=(N,).
            The class of each sample.
        reg: float. nonnegative.
            The regularization strength.
            NOTE: Ignore this until notebook instructions state otherwise.

        Returns:
        -----------
        float.
            The Hinge loss computed on the dataset `features`.
        ndarray. shape=(M,).
            The gradient of the Hinge loss with respect to each of the coefficients.
        float.
            The gradient of the Hinge loss with respect to the intercept.
        '''
        margin = y * (features @ self.c + self.intercept)
        per_sample = np.maximum(0.0, 1 - margin)  # hinge loss per sample
        hinge_loss = np.mean(per_sample) + np.square(np.linalg.norm(self.c)) * reg / 2  # hinge loss for n samples
        grad_c = (-1 / features.shape[0]) * (features.T @ (y * (margin < 1))) + reg * self.c
        grad_int = -np.mean(y * (margin < 1))

        return hinge_loss, grad_c, grad_int

    def predict(self, features):
        '''Classify -1 if plugging in a data sample in the hyperplane equation returns a negative value, and +1 if the
        equation returns a positive value for the current sample.

        Parameters:
        -----------
        features: ndarray. shape=(N, M).
            Data whose class we are predicting.

        Returns:
        -----------
        ndarray. shape=(N,).
            Predicted class for each sample — values may either be -1 or +1.
        '''
        pred = np.sign(features @ self.c + self.intercept)  # returns 0,-1,1
        pred = np.where(pred == 0, 1, pred)  # map 0 vals to 1
        return pred.astype(int)

    def train(self, features, y, reg=0, n_iter=1000, lr=0.001):
        '''Train the SVM to learn the optimal maximum hyperplane that separates training `features` by class. Learns
        using gradient descent and minimizing the Hinge loss.

        Parameters:
        -----------
        features: ndarray. shape=(N, M).
            Training dataset.
        y: ndarray. shape=(N,).
            Training dataset class labels. Assumes each label is either -1 or +1.
        reg: float. nonnegative.
            The regularization strength.
            NOTE: Ignore this until notebook instructions state otherwise.
        n_iter: int. positive.
            Number of iterations to run gradient descent.
        lr: float. positive.
            Learning rate to use in gradient descent.


        Returns:
        -----------
        list. len=n_iter
            The Hinge loss history computed on every iteration of gradient descent.
        None (initially). ndarray (later on).
            The indices of the support vectors.
        '''
        self.intercept = 0.0
        self.c = np.random.uniform(low=-.001, high=.001, size=(features.shape[1],))  # M samples between -.001,.001
        loss = []  # hinge loss per iteration

        # gradient descent
        for _ in range(n_iter):
            h_loss, grad_c, grad_int = self.hinge_loss(features, y, reg=reg)
            loss.append(h_loss)  # add to loss list

            # update c and intercept
            self.c = self.c - lr * grad_c
            self.intercept = self.intercept - lr * grad_int

        # find sup vecs
        margins = (y * (features @ self.c + self.intercept))
        sup_vecs = np.where(margins <= 1)[0]
        return loss, sup_vecs

    def hyperplane_2d(self, x_min=-5, x_max=15, n_sample_pts=50):
        '''Evaluates x and y values on the 2D optimal separating hyperplane (i.e. line).
        The hyperplane coefficients `c` correspond to an equation of form c1*x + c2*y + intercept = 0, but here we want
        to convert to y=m*x + b format.

        Parameters:
        -----------
        x_min: float.
            Minimum x value at which we should evaluate the SVM hyperplane line.
        x_max: float.
            Maximum x value at which we should evaluate the SVM hyperplane line.
        n_sample_pts: int.
            Number of sample points between `x_min` and `x_max` at which we should evaluate the SVM
            hyperplane line.

        Returns:
        -----------
        ndarray. shape=(n_sample_pts,).
            x sample points on SVM hyperplane
        ndarray. shape=(n_sample_pts,).
            y sample points on SVM hyperplane
        '''
        x_samples = np.linspace(x_min, x_max, n_sample_pts)
        # y = (c1 x + intercept) / -c2
        y = (self.c[0] * x_samples + self.intercept) / -self.c[1]
        return x_samples, y

    def compute_onesided_margin(self):
        '''Computes the one-sided SVM margin — the distance between between the optimal hyperplane and one of the
        gutters.

        Returns:
        -----------
        float.
            The one-sided margin, aka the distance between between the optimal hyperplane and one of the gutters.
        '''
        return 1 / np.linalg.norm(self.c)

    def gutter_offset(self):
        '''Computes the offset between the y values on the optimal separating hyperplane and the gutters on either side
        that flank samples belonging to the -1 class ("- sample gutter") and belonging to the +1 class
        ("+ sample gutter").

        (This is provided to you and should not require modification)

        Returns:
        -----------
        float.
            y offset between the hyperplane and either gutter.
        '''
        slope = -self.c[0] / self.c[1]
        offset = np.sqrt(1 + slope ** 2) * self.compute_onesided_margin()
        return offset


class SVMEnsemble(Classifier):
    '''An ensemble of support vector machine classifiers to handle multi-class classification problems'''

    def __init__(self, C):
        '''SVM Ensemble constructor

        Parameters:
        -----------
        C: int. Number of classes.
        '''
        super().__init__(C)
        self.svms = [SVM() for _ in range(C)]  # init C SVM models

    def recode_classes(self, y, class_ind):
        '''Recodes the class labels for each sample `y` to:
        - +1 for the current SVM preferred class `class_ind`.
        - -1 for all other classes.

        Parameters:
        -----------
        y: ndarray. shape=(N,).
            The class labels for the data samples.
        class_ind: int.
            The current SVM preferred class for which we are recoding the labels.

        Returns:
        -----------
        ndarray. shape=(N,).
            The class labels for the data samples recoded for the SVM that prefers class `class_ind`.

        '''
        return np.where(y == class_ind, 1, -1)  # 1 if class ind, -1 if not

    def train(self, features, y, reg=0, n_iter=1000, lr=0.001):
        '''Trains the SVM Ensemble

        Parameters:
        -----------
        features: ndarray. shape=(N, M).
            Training dataset.
        y: ndarray. shape=(N,).
            Training dataset class labels. Labels are nonnegative ints.
        reg: float. nonnegative.
            The regularization strength.
        n_iter: int. positive.
            Number of iterations to run gradient descent.
        lr: float. positive.
            Learning rate to use in gradient descent.
        '''
        loss = [] # 2d array of loss per svm
        sup_vecs = [] # 2d array of sup vec per svm
        for i, svm in enumerate(self.svms):
            l, s = svm.train(features, self.recode_classes(y, i), reg=reg, n_iter=n_iter, lr=lr)
            loss.append(l)
            sup_vecs.append(s)

        return loss, sup_vecs

    def predict(self, features):
        '''Predicts the class for each sample in `features` according to the SVM ensemble. The strategy is to compute
        the scores for each fitted SVM and sample. The predicted class for a particular sample is the preferred class
        of the SVM that achieves the highest score.

        Parameters:
        -----------
        features: ndarray. shape=(N, M).
            Training dataset.

        Returns:
        -----------
        ndarray. shape=(N,).
            Predicted class for each sample — values are nonnegative ints between 0 and C-1.
        '''
        scores = np.array([svm.compute_scores(features) for svm in self.svms])
        return np.argmax(scores, axis=0)  # get max pred score on svms for the samples
