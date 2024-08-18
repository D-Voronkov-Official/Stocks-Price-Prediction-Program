""""
    Model file contains:
        StocksPredictionModel - class that implements LSTM model
    
    Raises:
        Value error - if the integer arguments are not numbers or 
            model type is not the instance of ModelType class
"""

from numbers import Number
import torch
import torch.nn as nn
from type_enums.ModelType import ModelType as mt


class StocksPredictionModel(nn.Module):
    """
    Let's break all the parameters down:
        days_to_predict {integer} - how many days in the future we will want to predict on
        hidden size {integer} - number that will start from in the hidden state
        num_layers {integer} - how deep our model is. In our case - always 1
        modelType {ModelType} - variable of Enum type,
            determines whether the model will be simple or complex
        columns_amount {integer} (optional) - how many features our model will have (in our case - always 16)

        methods:
            __init__ - class constructor
            forward - predicting future stock prices
    """

    def __init__(
        self,
        days_to_predict,
        hidden_size,
        num_layers,
        batch_first=True,
        columns_amount=16,
        modelType=mt.ComplexModel,
        seed_number=None,
        standardized=True,
        lr=0.001,
    ):

        if not isinstance(modelType, mt):
            raise ValueError("Model type must have enum type!")

        super().__init__()
        self.kwargs = {
            "days_to_predict": days_to_predict,
            "columns_amount": columns_amount,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": batch_first,
            "modelType": modelType,
            "standardized": standardized,
        }
        self.days_to_predict = days_to_predict
        self.input_size = columns_amount
        self.standardized = standardized
        if seed_number is not None and isinstance(seed_number, Number):
            torch.manual_seed(seed_number)
        self.model_type = modelType
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.custom_ltsm = nn.LSTM(
            input_size=columns_amount,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        match self.model_type:

            case mt.ComplexModel:
                self.linear_sequence = nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, days_to_predict),
                )

            case mt.SimpleModel:
                self.linear_sequence = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, days_to_predict),
                )

            case mt.ExperimentalModel:
                self.linear_sequence = nn.Sequential(
                    nn.Linear(hidden_size, 90),
                    nn.LeakyReLU(),
                    nn.Linear(90, 90),
                    nn.LeakyReLU(),
                    nn.Linear(90, days_to_predict),
                )

            case _:
                raise ValueError("Unknown model type!")

    # Basically forward function is what we should do with data
    def forward(self, data):
        """
        Forward function defines how the model should work with data
        In our case it takes one parameter - stocks data
        In our use case data is equivalent to X_train and X_test
        It returns the output which later will be comapred with y_test and y_train

        Arguments:
            data {tensor} - data on which we want to predict future values
        """
        data = data.to(self.device)
        h0 = torch.zeros(
            self.num_layers, data.size(0), self.hidden_size
        ).requires_grad_()

        c0 = torch.zeros(
            self.num_layers, data.size(0), self.hidden_size
        ).requires_grad_()

        h0 = h0.to(self.device)

        c0 = c0.to(self.device)
        # LSTM returns output, hidden state and cell state state.
        # In our model we will be working only with hidden state
        _, (hidden_state, _) = self.custom_ltsm(data, (h0, c0))
        hidden_state = hidden_state.view(-1, self.hidden_size)
        hidden_state = hidden_state.to(self.device)

        return self.linear_sequence(hidden_state)
