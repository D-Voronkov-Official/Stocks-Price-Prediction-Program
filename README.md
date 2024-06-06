How to reproduce the environment:

1) Download the project from GitHub
2) In the command prompt go to folder with environment.yml file
3) Use command: 
conda env create -p |"PATH TO THE DESIRED FOLDER"| --file=environment.yml
NOTE: You can use -n argument instead of -p if you want yor project to live in the default conda directory



Project Structure:

1) Custom LSTM Model folder

* Files:

    a) Stocks Predictions Model class (blueprint) - Model.ipynb
    b) Data Set which gathers all the .csv files in the Data folder and split all the files into training and testing sets - StocksDataSet.ipynb
    c) Functions which trains the model with data set - Training Model.ipynb

* Folders:

    a) Models - Contains different trained models (which are trained with Training Model.ipynb
    b) Data - .csv files with stocks data for model training. Download functions can be found in "Machine Learning Data Gathering" folder, which resides inside the root of the project.

2) Machine Learning Data Gathering

* Folders:

a) Scripts:
    i) Getting Data For Machine Learning Model - scripts that uses yfinance framework to download latest stocks data
    ii) Moving Average + Bollinger bands.ipynb - additional functions that calculates useful stocks data, such as RSI, ATR, Bollinger Bands + plots stocks data with matplotlib framework.


Frameworks used:
    PyTorch (to build machine learning model that will predict stocks future price), 
    Pandas (for working with .csv files), 
    NumPy (to calculate stock related data, such as RSI, RTS and Moving Average), MatPlotlib (To draw stocks-related graphics), 
    yfinance - to gather up-to-date stocks data,
    matplotlib - to plot the stocks data,
    scikitlearn - for data normalization

