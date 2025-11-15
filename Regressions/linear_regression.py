'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Andrew Welling
CS 252: Mathematical Data Analysis Visualization
Spring 2025
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import analysis

class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable (true values) being predicted by linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's lstsq function.
            'normal': Use normal equations.
            'qr': Use QR factorization
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        if type(ind_vars) is not list:
            self.ind_vars = [ind_vars]

        x = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])
        self.A = np.hstack([np.ones((len(x), 1)), x]) # create a matrx
        if method=='scipy':
            c = self.linear_regression_scipy(self.A,self.y)
            self.intercept = c[0,0]
            if len(self.ind_vars) == 1: # we still want slope to be 2d
                self.slope = c[1].reshape(-1, 1)
            else:
                self.slope = c[1:]

        if method=='normal':
            c = self.linear_regression_normal(self.A,self.y)
            self.intercept = c[0,0]
            if len(self.ind_vars) == 1: # we still want slope to be 2d
                self.slope = c[1].reshape(-1, 1)
            else:
                self.slope = c[1:]
        if method=='qr':
            c = self.linear_regression_qr(self.A,self.y)
            self.intercept = c[0,0]
            if len(self.ind_vars) == 1: # we still want slope to be 2d
                self.slope = c[1].reshape(-1, 1)
            else:
                self.slope = c[1:]

        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.compute_residuals(y_pred)
        self.mse = self.compute_mse()

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        c, residuals, rank, s = scipy.linalg.lstsq(A, y)
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        A_t = np.matrix_transpose(A)
        c = np.linalg.inv(A_t @ A) @ A_t @ y
        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        Q,R = self.qr_decomposition(A)
        c = scipy.linalg.solve_triangular(R,np.transpose(Q) @ y)
        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        Q = np.zeros_like(A)
        for i in range(A.shape[1]): # col in A
            Acol = A[:, i].copy() # want this column to be orthog to cols in 0,i in Q
            temp_col = Acol

            if i > 0:
                for j in range(i): # projection
                    dot_product = np.dot(Acol, Q[:, j])  # proj
                    temp_col -= dot_product * Q[:, j]  # subtract

            temp_col /= (scipy.linalg.norm(temp_col)) # normalize

            Q[:,i] = temp_col # place into Q
        R = np.transpose(Q) @ A
        return Q,R

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        '''
        if X is None:
            A = self.A
        else:
            if self.p > 1:
                poly_mat = self.make_polynomial_matrix(X,self.p)
                A = np.hstack([np.ones((len(X), 1)), poly_mat])
            else:
                A = np.hstack([np.ones((len(X), 1)), X])
        return np.matmul(A,np.vstack([self.intercept,self.slope]))

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        return 1 - (np.sum((self.y - y_pred)**2) / np.sum((self.y - np.mean(self.y))**2))

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        return self.y - y_pred

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error
        '''
        return np.mean(self.residuals ** 2)

    def scatter(self, ind_var, dep_var, title=""):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot
        '''
        an = analysis.Analysis(self.data)

        x,y = an.scatter(ind_var, dep_var, title)
        line_x = np.linspace(start=np.min(x),stop=np.max(x),num=1000)
        if self.p == 1:
            line_y = np.squeeze(self.intercept + line_x * self.slope)
        else:
            line_x_poly = self.make_polynomial_matrix(line_x.reshape(-1, 1),self.p) # make poly matrix out of x line
            line_y = np.squeeze(self.intercept + line_x_poly @ self.slope)
        plt.plot(line_x, line_y, color='g')

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.
        '''
        an = analysis.Analysis(self.data)
        selected_data = self.data.select_data(headers=data_vars)
        fig, axes = an.pair_plot(data_vars,fig_sz)

        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                x = selected_data[:, i]
                y = selected_data[:, j]
                ax = axes[i,j]
                self.linear_regression(data_vars[i],data_vars[j])
                line_x = np.linspace(start=np.min(x), stop=np.max(x), num=50)
                line_y = np.squeeze(self.intercept + line_x * self.slope)
                ax.title.set_text(f"R2: {round(self.R2, 2)}")
                ax.plot(line_x, line_y, color='g')

            if hists_on_diag: # handle histograms
                numVars = len(data_vars)
                axes[i, i].remove()
                axes[i, i] = fig.add_subplot(numVars, numVars, i * numVars + i + 1)
                if i < numVars - 1:
                    axes[i, i].set_xticks([])
                else:
                    axes[i, i].set_xlabel(data_vars[i])
                if i > 0:
                    axes[i, i].set_yticks([])
                else:
                    axes[i, i].set_ylabel(data_vars[i])

                axes[i, i].hist(selected_data[:, i], bins=20, color='lightblue')
        # adjust spacing
        fig.subplots_adjust(top=1, wspace=5, hspace=1)
        fig.tight_layout(pad=2.0)




    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.
        '''
        poly_mat = []
        for var in A:
            row = [0] * p
            for j in range(1,p+1):
                row[j-1] = var[0] ** j
            poly_mat.append(row)
        return np.array(poly_mat).astype(float)


    def poly_regression(self, ind_var, dep_var, p, method='normal'):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1st).
        method: str. Least squares solver method to use.
            Supported options: 'normal', 'scipy', 'qr' (to be added later)
        '''
        self.ind_vars = ind_var
        self.dep_var = dep_var
        self.p = p
        if type(ind_var) is not list:
            self.ind_vars = [ind_var]

        x = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])
        poly_mat = self.make_polynomial_matrix(x,p) # convert x data into a polymatrix
        self.A = np.hstack([np.ones((len(x), 1)), poly_mat])  # create A matrix
        if method == 'scipy':
            c = self.linear_regression_scipy(self.A, self.y)
            self.intercept = c[0, 0]
            self.slope = c[1:]
            if len(self.slope) == 1:  # we still want slope to be 2d
                self.slope = c[1].reshape(-1, 1)

        if method == 'normal':
            c = self.linear_regression_normal(self.A, self.y)
            self.intercept = c[0, 0]
            self.slope = c[1:]
            if len(self.slope) == 1:  # we still want slope to be 2d
                self.slope = c[1].reshape(-1, 1)
        if method == 'qr':
            c = self.linear_regression_qr(self.A, self.y)
            self.intercept = c[0, 0]
            self.slope = c[1:]
            if len(self.slope) == 1:  # we still want slope to be 2d
                self.slope = c[1].reshape(-1, 1)

        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred)
        self.residuals = self.compute_residuals(y_pred)
        self.mse = self.compute_mse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p