# Stocks Predictions Program
## How to reproduce the environment:

1) Download the project from GitHub
2) In the command prompt go to folder with environment.yml file
3) Use command: 
conda env create -p |*"PATH TO THE DESIRED FOLDER"*| --file=environment.yml
**NOTE:** You can use -n argument instead of -p if you want yor project to live in the default conda directory



# **Project Structure:**

## 1) Custom LSTM Model folder

### * **Files**:

    a) * Stocks Predictions Model class * (blueprint) - Model.ipynb
    b) Data Set which gathers all the .csv files in the Data folder and split all the files into training and testing sets - StocksDataSet.ipynb
    c) Functions which trains the model with data set - Training Model.ipynb

### * **Folders**:

    a) Models - Contains different trained models (which are trained with Training Model.ipynb
    b) Data - .csv files with stocks data for model training. Download functions can be found in "Machine Learning Data Gathering" folder, which resides inside the root of the project.

## 2) Machine Learning Data Gathering

### * **Folders:**

a) Scripts:

    a) Getting Data For Machine Learning Model - scripts that uses yfinance framework to download latest stocks data
    b) Moving Average + Bollinger bands.ipynb - additional functions that calculates useful stocks data, such as RSI, ATR, Bollinger Bands + plots stocks data with matplotlib framework.


# Frameworks used:
    a) PyTorch (to build machine learning model that will predict stocks future price), 
    b) Pandas (for working with .csv files), 
    c) NumPy (to calculate stock related data, such as RSI, RTS and Moving Average), MatPlotlib (To draw stocks-related graphics), 
    d) yfinance - to gather up-to-date stocks data,
    e) matplotlib - to plot the stocks data,
    f) scikitlearn - for data normalization (since PyTorch does not provide suitable tools for normalization)


# Clear the output and metadata:

    Since .gitattributes is already presented, we need to edit local git config. This can be achieved with the following command:
        git config --local --edit
    after that we need to add this line to the config:
        [filter "strip-notebook-output"]
        clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"

# How to work with the project:
1) Reproduce the environment 
                (steps can be found in the correlated section in the Readme file at the start of the document)
2) Download the training data 
                (by running **Getting Data For Machine Learning Model.ipynb** 
                file which is located in **Machine Learning Data Gathering/Scripts** folder)
3) Run the training model file 
                (**Custom LSTM model/Training Model.ipynb**)
4) Test the training model 
                (**Custom LSTM model/models_testing/testing_models_script.ipynb**) (**NOTE:** Currently in development)
