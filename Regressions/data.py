'''data.py
Reads CSV files, stores data, access/filter data by variable name
Andrew Welling
CS 251/2: Data Analysis and Visualization
Spring 2025
'''
import numpy as np

class Data:
    '''Represents data read in from .csv files
    '''
    def __init__(self, filepath=None, headers=None, data=None, header2col=None, cats2levels=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        cats2levels: Python dictionary or None.
                Maps each categorical variable header (var str name) to a list of the unique levels (strings)
                Example:

                For a CSV file that looks like:

                letter,number,greeting
                categorical,categorical,categorical
                a,1,hi
                b,2,hi
                c,2,hi

                cats2levels looks like (key -> value):
                'letter' -> ['a', 'b', 'c']
                'number' -> ['1', '2']
                'greeting' -> ['hi']

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - cats2levels
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col
        self.cats2levels = cats2levels
        if filepath is not None:
            self.read(filepath)

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called `self.data` at the end
        (think of this as a 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
        '''
        self.filepath = filepath
        self.header2col = {}
        self.cats2levels = {}
        self.data = []
        self.headers = []
        # read the file
        with open(filepath, 'r') as f:
            # get the first and second row to get headers and cats
            head_row = f.readline()
            type_row = f.readline()
            # init header2col and cats2levels, also retrieve data types
            headers = head_row.split(',')
            headers = [h.strip() for h in headers]
            types = type_row.split(',')
            types = [t.strip() for t in types]
            data_types = []
            for dt in types: # cleanse data types
                d = dt.strip()
                if d in ['numeric', 'categorical']:
                    data_types.append(d)
                else:
                    if d not in ['string', 'date']: # raise exception on invalid
                        print(d)
                        raise Exception("Invalid data types identified in the 2nd row")

            # init headers, head2col, and cast2 levels properly
            # also check to make sure data types and headers are valid
            count = 0 # counts needed to match headers with proper types
                      # each missed type increments it by 1, so the indices of
                      # header stay correct
            for n in range(len(headers)):
                # init
                if types[n] in data_types:
                    self.headers.append(headers[n])
                    self.header2col[headers[n]] = n - count
                    if types[n] == 'categorical':
                        self.cats2levels[headers[n]] = []
                else: count += 1


            for row in f:
                data_row = row.split(',')
                data_row = [d.strip() for d in data_row]
                new_data = []
                for i in range(len(data_row)):
                    if types[i] == 'numeric': # for floats
                        try:
                            value = float(data_row[i])
                            if data_row[i] == '':
                                # on missing case
                                new_data.append(np.nan)
                            else:
                                new_data.append(value)
                        except ValueError:
                            # treat as invalid
                            new_data.append(np.nan)

                    elif types[i] == 'categorical': # for ints
                        try:
                            value = int(data_row[i])
                            if data_row[i] == '':
                                # on missing case
                                if "Missing" not in self.cats2levels[headers[i]]:
                                    self.cats2levels[headers[i]].append("Missing")
                                new_data.append(self.cats2levels[headers[i]].index("Missing"))
                            else:
                                if value not in self.cats2levels[headers[i]]:
                                    self.cats2levels[headers[i]].append(value)
                                new_data.append(self.cats2levels[headers[i]].index(value))
                        except ValueError:
                            # treat as string
                            value = str(data_row[i])
                            if value == '':
                                if "Missing" not in self.cats2levels[headers[i]]:
                                    self.cats2levels[headers[i]].append("Missing")
                                new_data.append(self.cats2levels[headers[i]].index("Missing"))
                            else:
                                if value not in self.cats2levels[headers[i]]:
                                    self.cats2levels[headers[i]].append(value)
                                new_data.append(self.cats2levels[headers[i]].index(value))

                # app row to the data list
                self.data.append(new_data)


        self.data = np.array(self.data) # turn into an np any dimensional array

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's called to determine
        what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatt2ed string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.

        NOTE: It is fine to print out int-coded categorical variables (no extra work compared to printing out numeric data).
        Printing out the categorical variables with string levels would be a small extension.
        '''
        to_str = ""
        to_str += self.filepath +  ' (' + str(self.get_num_samples()) + 'x' + str(self.get_num_dims()) + ')' + '\n'
        to_str += 'Headers:' + '\n' + "  ".join(self.headers) + '\n'
        to_str += "-" * len(to_str) + '\n'
        display_count = min(5, len(self.data))
        if display_count != len(self.data):
            to_str += "Showing first " + str(display_count) + "/" + str(len(self.data)) + " rows\n"
        to_str += "\n".join("  ".join(map(str, row)) for row in self.data[:display_count])
        to_str += '\n'
        return to_str

    def get_headers(self):
        '''Get list of header names (all variables)

        Returns:
        -----------
        Python list of str.
        '''
        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_cat_level_mappings(self):
        '''Get method for mapping between categorical variable names and a list of the respective unique level strings.

        Returns:
        -----------
        Python dictionary. str -> list of str
        '''
        return self.cats2levels

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return self.data.shape[1]

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.data.shape[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers` list.
        '''
        return [self.header2col[header] for header in headers]

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself. This can be accomplished with numpy's copy
            function.
        '''
        return self.data.copy()

    def head(self):
        '''Return the 1st five data samples (all variables)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        return self.data[:min(5,len(self.data)) , :]

    def tail(self):
        '''Return the last five data samples (all variables)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        return self.data[-min(5, len(self.data)):, :]

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.
        '''
        self.data = self.data[start_row:end_row, :]

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified by the `rows` list.

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return column #2 of self.data.
        If rows is not [] (say =[0, 2, 5]), then we do the same thing, but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data
        rows: Python list of ints OR NumPy ndarray of ints. Indices of subset of data samples to select.
            Empty list [] means take all rows.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.
        '''
        cols = self.get_header_indices(headers)
        if len(rows) == 0:
            return self.data[:, cols]
        else:
            if len(rows) <= self.data.shape[0]:
                return self.data[np.ix_(rows, cols)]
            else: # return all data on invalid rows
                return self.data[:, cols]