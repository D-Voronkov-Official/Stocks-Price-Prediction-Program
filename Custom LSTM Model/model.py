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
        num_classes {integer} - how many features will our model have
        hidden size {integer} - how many features in the hidden state
                        (in our case - always 1, because we are looking for closing price,
                            which is a 1 column otput (1 feature))
        num_layers {integer} - how deep our model is. In our case - always 1
        modelType {ModelType} - variable of Enum type, 
            determines whether the model will be simple or complex

        methods:
            __init__ - class constructor
            forward - predicting future stock prices 
    """

    def __init__(
        self,
        num_classes,
        hidden_size,
        num_layers,
        batch_first=True,
        input_size=16,
        modelType=mt.ComplexModel,
        seed_number=None,
    ):

        if not isinstance(modelType, mt):
            raise ValueError("Model type must have enum type!")

        super().__init__()
        self.kwargs = {
            "num_classes": num_classes,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": batch_first,
            "modelType": modelType,
        }

        self.input_size = input_size
        if seed_number is not None and isinstance(seed_number, Number):
            torch.manual_seed(seed_number)
        self.model_type = modelType
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.custom_ltsm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        match self.model_type:

            case mt.ComplexModel:
                self.linear_sequence = nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes),
                )

            case mt.SimpleModel:
                self.linear_sequence = nn.Sequential(
                    nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, num_classes)
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
        h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).requires_grad_()

        c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).requires_grad_()
        h0 = h0.to(self.device)

        c0 = c0.to(self.device)
        # LSTM returns output, hidden state and cell state state.
        # In our model we will be working only with hidden state
        _, (hidden_state, _) = self.custom_ltsm(data, (h0, c0))
        hidden_state = hidden_state.view(-1, self.hidden_size)
        hidden_state = hidden_state.to(self.device)

        return self.linear_sequence(hidden_state)
