""""
    This file contains 1 class that is needed to work with 
    multiple .csv files in order to train our model on
    different stocks.

    It reads all the csv files from the path (including nested folders)
    split them and returns data that can be used for model training
"""

import glob
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from type_enums.SplitType import SplitType as st


class StocksDataSet(Dataset):
    """
        This class represents custom implementation of dataset
        It basically takes path to the folder and get all the .csv stock files from it
        Then it splits the files into the training and testing sets and returns them if we call
        __getitem__ method

        methods:
            __init__ - constructor
            __len__ - returns the amount of .csv files in the folder
            prepare_data_for_mlmodel - split the X and y into adjustable timeframes
            experimental_preparation - mostly the same as prepare_data_for_mlmodel 
                but with different algorithm of splitting
            split_data - splitting the data into training and testing sets

    """

    def __init__(
        self, root_directory, preparation_type=st.CustomSplit, split_percentage=0.80
    ):
        """Class constructor

        Args:
            root_directory (String): path to the folder with .csv stock files
            preparation_type (SplitType (Enum), optional): How the data will be splitted.
               Defaults to st.CustomSplit.
            split_percentage (float, optional): split percentage can be adjusted by the user. 
                Defaults to 0.80.

        Raises:
            ValueError: If split type is not instance of CustomSplit
        """

        if not isinstance(preparation_type, st):
            raise ValueError("Split type must have enum type!")
        self.prep_type = preparation_type
        self.split_percentage = split_percentage

        self.root = root_directory
        self.stocks_files = glob.glob(f"{self.root}/**/*.csv", recursive=True)
        self.stand_scaler = StandardScaler()
        self.mm_scaler = MinMaxScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.stocks_files)

    def prepare_data_for_mlmodel(
        self, input_data, output_data, steps_for_input, steps_for_output
    ):
        """ Custom Preparation of data for machine learning model

        Args:
            input_data (dataframe): dataframe with features that will help us to predict data
            output_data (dataframe): results data for training
            steps_for_input (int): how much days we are getting in order to predict
            steps_for_output (int): how much values will be predicted

        Returns:
            np.array : returns splitted X and y arrays for machine learning model
        """
        X, y = [], []
        total_steps = steps_for_input + steps_for_output

        rows_amount = len(input_data.index)
        if rows_amount % total_steps != 0:
            input_data = input_data.iloc[rows_amount % total_steps :]
            output_data = output_data.iloc[rows_amount % total_steps :]
            rows_amount = len(input_data.index)

        for i in range(0, rows_amount, total_steps):
            X.append(input_data[i : i + steps_for_input])
            y.append(
                output_data[
                    i + steps_for_input : (i + steps_for_input) + steps_for_output
                ]
            )
        return np.array(X), np.array(y)

    def split_data(self, X_full, y_full, split_percentage=0.80):
        """Splitting the X and y into training and testing datasets

        Args:
            X_full (np.array): All features excluding Close price. 
            y_full (np.array): Close price that is converted into np.array
            split_percentage (float, optional): how much data we are giving to the training set 
                Defaults to 0.80.

        Returns:
            torch.Tensor: returns 4 Tensors, where X and y's are 
                splitted into training and testing sets (2 training and 2 testing sets accordingly)
        """
        total_data = len(X_full)

        test_split = round(split_percentage * total_data)

        X_train = torch.Tensor(X_full[:test_split])
        X_test = torch.Tensor(X_full[test_split:])

        y_train = torch.Tensor(y_full[:test_split])
        y_test = torch.Tensor(y_full[test_split:])

        if X_test.size(dim=0) == 0 or y_test.size(dim=0) == 0:
            X_train = torch.Tensor(X_full[:-1, :, :])
            X_test = torch.Tensor(X_full[-1:, :, :])
            y_train = torch.Tensor(y_full[:-1, :, :])
            y_test = torch.Tensor(y_full[-1:, :, :])

        X_train = torch.Tensor(
            self.stand_scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
        )
        X_test = torch.Tensor(
            self.stand_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
                X_test.shape
            )
        )

        y_train = torch.Tensor(
            self.mm_scaler.fit_transform(
                y_train.reshape(-1, y_train.shape[-1])
            ).reshape(y_train.shape)
        )
        y_test = torch.Tensor(
            self.mm_scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(
                y_test.shape
            )
        )

        return X_train, X_test, y_train, y_test

    def experimental_preparation(
        self,
        input_data,
        output_data,
        steps_for_input,
        steps_for_output
    ):
        X, y = [], []

        for i in range(len(input_data)):
            end_x = i + steps_for_input

            output_x = end_x + steps_for_output - 1
            if output_x > len(input_data):
                break

            seq_x, seq_y = input_data[i:end_x], output_data[end_x - 1 : output_x]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)

    def __getitem__(self, stock_index):
        """Gets the single .csv file from the folder and prepares it for the ML model 

        Args:
            stock_index (string): path to the csv file

        Raises:
            ValueError: Raised if preparation_type is not ExperimentalSplit or CustomSplit

        Returns:
            Tensor: 4 tensors, 2 for training and 2 for testing 
        """
        data = pd.read_csv(
            self.stocks_files[stock_index], index_col="Date", parse_dates=True
        )

        if len(data.index) > 7000:
            data = data[5000:]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        close = data["Close"]
        related_features = data.drop("Close", axis=1)

        match self.prep_type:
            case st.ExperimentalSplit:
                X_calc, y_calc = self.experimental_preparation(
                    related_features, close, 90, 30
                )
            case st.CustomSplit:
                X_calc, y_calc = self.prepare_data_for_mlmodel(
                    related_features, close, 90, 30
                )
            case _:
                raise ValueError("Unknown preparation type!")

        return self.split_data(X_calc, y_calc, self.split_percentage)
