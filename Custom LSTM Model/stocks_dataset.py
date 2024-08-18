""""
    This file contains 1 class that is needed to work with 
    multiple .csv files in order to train our model on
    different stocks.

    It reads all the csv files from the path (including nested folders)
    split them and returns data that can be used for model training
"""

import glob
import os
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
        self,
        root_directory,
        preparation_type=st.CustomSplit,
        split_percentage=0.80,
        standardized=True,
        days_to_look=90,
        days_result=30,
    ):
        """Class constructor

        Args:
            root_directory (String): path to the folder with .csv stock files
            preparation_type (SplitType (Enum), optional): How the data will be splitted.
               Defaults to st.CustomSplit.
            split_percentage (float, optional): split percentage can be adjusted by the user.
                Defaults to 0.80.
            standardized (boolean, optional): flag to decide whether the data should be standardized

        Raises:
            ValueError: If split type is not instance of CustomSplit
        """

        if not isinstance(preparation_type, st):
            raise ValueError("Split type must have enum type!")
        self.prep_type = preparation_type
        self.split_percentage = split_percentage
        self.standardized = standardized
        self.root = root_directory
        self.stocks_files = glob.glob(f"{self.root}/**/*.csv", recursive=True)
        self.stand_scaler = StandardScaler()
        self.mm_scaler = MinMaxScaler()

        self.days_to_look = days_to_look
        self.days_result = days_result
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.stocks_files)

    def prepare_data_for_mlmodel(
        self, input_data, output_data, steps_for_input, steps_for_output
    ):
        """Custom Preparation of data for machine learning model

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
        if self.prep_type == st.QuarterSplit:
            X_train = torch.Tensor(X_full[:-2])
            X_test = torch.Tensor(X_full[-2:])
            y_train = torch.Tensor(y_full[:-2])
            y_test = torch.Tensor(y_full[-2:])

            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

        else:
            total_data = len(X_full)

            test_split = round(split_percentage * total_data)

            X_train = torch.Tensor(X_full[:test_split])
            X_test = torch.Tensor(X_full[test_split:])

            y_train = torch.Tensor(y_full[:test_split])
            y_test = torch.Tensor(y_full[test_split:])

        if (
            X_test.size(dim=0) == 0
            or y_test.size(dim=0) == 0
            or X_train.size(dim=0) == 1
        ):
            raise ValueError("Can't split tensor with current split percentage")

        if self.standardized:
            X_train = torch.Tensor(
                self.stand_scaler.fit_transform(
                    X_train.reshape(-1, X_train.shape[-1])
                ).reshape(X_train.shape)
            )
            X_test = torch.Tensor(
                self.stand_scaler.transform(
                    X_test.reshape(-1, X_test.shape[-1])
                ).reshape(X_test.shape)
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
        self, input_data, output_data, steps_for_input, steps_for_output
    ):
        """Preparing the data for splitting before giving it to the model

        Args:
            input_data (dataframe): dataframe with features, based on which we want to predict
            output_data (dataframe): dataframe with feature that we want to predict
            steps_for_input (integer): how many days we will look in the past to predict
            steps_for_output (integer): how many days in the future we will predict

        Returns:
            numpy array: returns 2 numpy arrays - first one - with the data on which model will
                                                                                make prediction
                and second one with the results on which we will make
                the assumption whether our mopdel predicts well enough to use it in real world
        """

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

    def split_quarterly(self, stocks_prices, close_price, earning_dates):
        """Splits the stocks data and closing price into 2 numpy array

        Args:
            stocks_prices (df): dataframe with all the necessary stocks data (except price)
            close_price (df): dataframe with closing price and related date
            earning_dates (df): dataframe with the stock earning prices (we will use date column from this df)

        Raises:
            ValueError: occurs when the function returns empty array. Exception is needed to prevent adding empty
            dataset into training data

        Returns:
            Numpy array: returns x and y numpy array, which will be splitted into training and testing datasets later
        """
        pd.options.mode.chained_assignment = None
        earning_dates["Earnings Date"] = earning_dates["Earnings Date"].values[::-1]

        X, y = [], []

        earning_dates["Earnings Date"] = pd.to_datetime(earning_dates["Earnings Date"])

        for i in range(0, len(earning_dates.index) - 2, 1):

            stocks_prices["Date"] = pd.to_datetime(stocks_prices["Date"]).dt.normalize()

            X_data = stocks_prices[
                (stocks_prices["Date"] > earning_dates.loc[i, "Earnings Date"])
                & (stocks_prices["Date"] < earning_dates.loc[i + 1, "Earnings Date"])
            ]
            close_data = close_price["Close"][
                close_price["Date"].between(
                    earning_dates.loc[i + 1, "Earnings Date"],
                    earning_dates.loc[i + 2, "Earnings Date"],
                )
            ]

            if len(X_data.index) < 60 or len(close_data.index) < 60:
                continue
            X_full = X_data[-60:]
            X_full.drop("Date", axis=1, inplace=True)

            X.append(X_full)
            sliced_close = close_data[:60]
            sliced_final = pd.DataFrame({"Close": sliced_close})
            y.append(sliced_final)
        # If, by any chance, the array is empty - throw an exception
        if len(X) == 0:
            raise ValueError("Nothing was added, skip!")
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

        data = data.round(2)
        data.drop(["Open", "High", "Low"], inplace=True, axis=1)

        if data.shape[0] < 1000:
            raise ValueError("File skipped because it contains less than 1000 rows!")

        if self.prep_type == st.QuarterSplit:
            data = data[-3000:]
        else:
            data = data[-1000:]

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        if self.prep_type == st.QuarterSplit:
            data = data.reset_index()
            close = data[["Date", "Close"]]
        else:
            close = data["Close"]
        related_features = data.drop("Close", axis=1)
        match self.prep_type:
            case st.ExperimentalSplit:
                X_calc, y_calc = self.experimental_preparation(
                    related_features, close, self.days_to_look, self.days_result
                )
            case st.CustomSplit:
                X_calc, y_calc = self.prepare_data_for_mlmodel(
                    related_features, close, self.days_to_look, self.days_result
                )
            case st.QuarterSplit:
                earning_path = f"./Data/Earnings Dates/{os.path.basename(self.stocks_files[stock_index]).removesuffix('.csv')}_earnings.csv"

                if not os.path.isfile(earning_path):
                    raise ValueError(f"No earnings for the {earning_path} file, skip!")
                earnings_dates = pd.read_csv(earning_path)
                X_calc, y_calc = self.split_quarterly(
                    related_features, close, earning_dates=earnings_dates
                )
            case _:
                raise ValueError("Unknown preparation type!")

        X_tr, X_t, y_tr, y_t = self.split_data(X_calc, y_calc, self.split_percentage)

        return Ds_Data_Container(
            X_tr, X_t, y_tr, y_t, os.path.basename(self.stocks_files[stock_index])
        )


class Ds_Data_Container:
    """
    This class serves as a container that contains all the necessary data for training 
                                                        + current file name.
    That way we can easily track on what file we are working now.
    """

    def __init__(self, X_train, X_test, y_train, y_test, file_name) -> None:
        self.X_train = X_train
        self._X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.file_name = file_name

    def get_data(self):
        """Return all the trainig/testing data

        Returns:
            4 Tensors: 2 for the training + 2 for the testing
        """
        return self.X_train, self._X_test, self.y_train, self.y_test