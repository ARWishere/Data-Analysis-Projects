'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Andrew Welling
CS 251/2: Data Analysis and Visualization
Spring 2025
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Array-like (e.g. Python list or numpy array) of ints.
            Indices of data samples to restrict computation of min over, or over all indices if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        '''
        return np.min(self.data.select_data(headers, rows), axis=0)

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Array-like (e.g. Python list or numpy array) of ints.
            Indices of data samples to restrict computation of max over, or over all indices if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        '''
        return np.max(self.data.select_data(headers, rows), axis=0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Array-like (e.g. Python list or numpy array) of ints.
            Indices of data samples to restrict computation of min/max over, or over all indices if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        '''
        return self.min(headers,rows), self.max(headers,rows)

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Array-like (e.g. Python list or numpy array) of ints.
            Indices of data samples to restrict computation of mean over, or over all indices if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        '''
        relevant_data = self.data.select_data(headers, rows)
        if relevant_data.shape[0] == 0:
            return np.zeros(len(headers))

        return np.sum(relevant_data, axis=0) / relevant_data.shape[0]

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Array-like (e.g. Python list or numpy array) of ints.
            Indices of data samples to restrict computation of variance over, or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables
        '''
        relevant_data = self.data.select_data(headers, rows)
        if relevant_data.shape[0] <= 1:
            return np.zeros(len(headers))

        mean = self.mean(headers, rows)
        return np.sum((relevant_data-mean)**2, axis=0) / (relevant_data.shape[0] - 1) # sum((xi - mean)^2) / n-1
        pass


    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Array-like (e.g. Python list or numpy array) of ints.
            Indices of data samples to restrict computation of standard deviation over, or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables
        '''
        return np.sqrt(self.var(headers,rows))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and "y" variable in the dataset
        `dep_var`. Both `ind_var` and `dep_var` should be strings in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot
        '''
        if ind_var not in self.data.headers or dep_var not in self.data.headers:
            raise Exception("Invalid headers")
        else:
            plt.title(title)
            plt.xlabel(ind_var)
            plt.ylabel(dep_var)
            selected_data = self.data.select_data(headers=[ind_var, dep_var])
            plt.scatter(selected_data[:, 0], selected_data[:, 1], marker='o')
            return selected_data[:, 0], selected_data[:, 1]

    def line(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and "y" variable in the dataset
        `dep_var`. Both `ind_var` and `dep_var` should be strings in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot
        '''
        if ind_var not in self.data.headers or dep_var not in self.data.headers:
            raise Exception("Invalid headers")
        else:
            plt.title(title)
            plt.xlabel(ind_var)
            plt.ylabel(dep_var)
            selected_data = self.data.select_data(headers=[ind_var, dep_var])
            plt.plot(selected_data[:, 0], selected_data[:, 1], marker='o')
            return selected_data[0], selected_data[1]

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in `data_vars` in the
        x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        '''
        fig, axes = plt.subplots(nrows=len(data_vars), ncols=len(data_vars), figsize=fig_sz, sharex='col', sharey='row')
        fig.title = title
        plt.rcParams.update({'font.size': 5})
        selected_data = self.data.select_data(headers=data_vars)
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                if i == len(data_vars) - 1:
                    # last row
                    axes[i][j].set_xlabel(data_vars[j])
                if j == 0:
                    # first column
                    axes[i][j].set_ylabel(data_vars[i])
                axes[i][j].scatter(selected_data[:, i], selected_data[:, j],  s=1)# plot data
                axes[i][j].set_title = f"{data_vars[i]} vs {data_vars[j]}" # plot title

        return fig, axes
