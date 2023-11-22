import math
import numpy as np
import pandas as pd
import tensorflow as tf

class DataLoader():

    def __init__(self, filename, split, cols, output_idx):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.output_idx = output_idx

    def get_test_data(self, seq_len, normalise):

        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, self.output_idx]
        return x,y

    def get_train_data(self, seq_len, batch_size, normalise):

        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        # print(data_x.shape)
        # 把ndarray转换为tf.dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)

        return train_dataset, data_x, data_y

    def generate_train_batch(self, seq_len, batch_size, normalise):
        
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    # def _next_window(self, i, seq_len, normalise):
    #     window = self.data_train[i:i+seq_len]
    #     window = self.normalise_windows(window, single_window=True)[0] if normalise else window
    #     x = window[:-1]
    #     y = window[-1, self.output_idx]
    #     return x, y
    
    def _next_window(self, i, seq_len, normalise):
        """
        x可以标准化,但y不用
        """
        window = self.data_train[i: i+seq_len]
        x = window[:-1]
        y = window[-1, self.output_idx]
        x = self.normalise_windows(x, single_window=True)[0] if normalise else x
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                # if float(window[0, col_i]) == 0:
                #     normalised_col = [((float(p) / float(window[0, col_i] + 1e-9)) - 1) for p in window[:, col_i]]
                # else:
                #     normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                max_v = max(window[:, col_i])
                min_v = min(window[:, col_i])
                normalised_col = [((float(p) - min_v) / (max_v - min_v)) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T 
            normalised_data.append(normalised_window)
        return np.array(normalised_data)