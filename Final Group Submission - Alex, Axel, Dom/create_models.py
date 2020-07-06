'''
COEN 281
Final Project
Group 2
Alex Cherekdjian
Axel Perez
Dom Magdaluyo
'''
import pandas as pd
import numpy as np
import glob
import sys
import pathlib
import datetime
import csv
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor


data, X_train, X_test, Y_train, Y_test= None, None, None, None, None
airport_code = None

def concatenate_data(years):
    '''
    Concatenate a given number of csv files in the flight-data directory. Set global dataframe for the data.
    '''
    assert((years > 0)and(years<11))
    path = str(pathlib.Path().absolute()) + "/flight-data"
    
    # grab all csv files in that path
    all_files = glob.glob(path + "/*.csv")
    all_data = []
    intervals = 0
    for filename in all_files:
        if intervals > years:
            break
        temp = pd.read_csv(filename, index_col=None, header=0)
        all_data.append(temp)
        intervals+=1
    global data
    # set global data with concatenated data
    data = pd.concat(all_data, axis=0, ignore_index=True)
    

def preprocess():
    '''
    Preprocess flight data.
    '''
    
    # commented columns are the ones we are keeping for now
    dropped_columns = [
    # 'FL_DATE',
     'OP_CARRIER',
    'OP_CARRIER_FL_NUM',
    #'ORIGIN',
    #'DEST',
    #'CRS_DEP_TIME',
    'DEP_TIME',
    #'DEP_DELAY',
    'TAXI_OUT',
    'WHEELS_OFF',
    'WHEELS_ON',
    'TAXI_IN',
    #'CRS_ARR_TIME',
    'ARR_TIME',
    #'ARR_DELAY',
    'CANCELLED',
    'CANCELLATION_CODE',
    'DIVERTED',
    'CRS_ELAPSED_TIME',
    #'ACTUAL_ELAPSED_TIME',
    'AIR_TIME',
    'DISTANCE',
    'CARRIER_DELAY',
    'WEATHER_DELAY',
    'NAS_DELAY',
    'SECURITY_DELAY',
    'LATE_AIRCRAFT_DELAY',
    'Unnamed: 27'
    ]

    # drop columns that are unecessary and all rows with NA
    X_samples = data.drop(columns=dropped_columns)
    X_samples.dropna(inplace = True)

    # drop outliers with a total delay of over 150 minutes or less delay then negative 150 minutes (early)
    sum_delay = X_samples['ARR_DELAY'] + X_samples['DEP_DELAY']
    X_samples['TOTAL_DELAY'] = sum_delay
    less = X_samples[(X_samples.TOTAL_DELAY < -150)]
    great = X_samples[(X_samples.TOTAL_DELAY > 150)]
    X_samples.drop(great.index, inplace=True)
    X_samples.drop(less.index, inplace=True)
    # drop columns that were used for removing outliers
    X_samples = X_samples.drop(columns =['TOTAL_DELAY', 'ARR_DELAY', 'DEP_DELAY'])


    # get specific origin airport data to train the model if an origin airport is specified
    if len(sys.argv) == 3:
        X_samples = X_samples.loc[X_samples['ORIGIN'] == sys.argv[2]]
        airport_code = sys.argv[2]
    else:
        airport_code = "ALL"
    

    # create encoders for both origin and dest
    encoding = 0
    # save encoding in dictionary that can be used in GUI for decoding user input
    airports = {}

    origins = X_samples['ORIGIN']
    destinations = X_samples['DEST']
    encoded_orig = []
    for i in range(len(origins)):
        if origins.iloc[i] not in airports.keys():
            airports[origins.iloc[i]] = float(encoding)
            encoding += 1

        encoded_orig.append(airports[origins.iloc[i]])
    

    # encode airports that are only destinations as well
    encoded_dest = []
    for i in range(len(destinations)):
        if destinations.iloc[i] not in airports.keys():
            airports[destinations.iloc[i]] = float(encoding)
            encoding += 1

        encoded_dest.append(airports[destinations.iloc[i]])
    
    # seperate dates into bins of months of width of 4

    monthbin = []

    for dates in X_samples['FL_DATE']:
        d = dates.split('-')
        bins = 0
        if int(d[1]) <= 3:
            bins = 1
        elif int(d[1]) <= 6:
            bins = 2
        elif int(d[1]) <= 9:
            bins = 3
        elif int(d[1]) <= 12:
            bins = 4
        monthbin.append(bins)

    # drop columns and create columns with new encoded data
    X_samples = X_samples.drop(columns=['ORIGIN','DEST','FL_DATE'])
    X_samples['ORIGIN'] =encoded_orig
    X_samples['DEST'] = encoded_dest
    X_samples['FL_DATE'] = monthbin

    # getting y and x samples
    Y_samples = X_samples['ACTUAL_ELAPSED_TIME']
    X_samples = X_samples.drop(columns=['ACTUAL_ELAPSED_TIME'])

    global X_train
    global X_test
    global Y_train 
    global Y_test
  
    # splitting the training data into train and test, set them globally
    X_train, X_test, Y_train, Y_test = train_test_split(X_samples.values, Y_samples.values, test_size=0.25)
    

def poly_regression(reg_type):
    '''
    Run polynomial regression with a specified ridge regression type on several n and alpha values.
    '''
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # finding best n and alpha value
        score_min = sys.maxsize
        for pol_order in range(2, 3):
            for alpha in range(1, 20, 2):
                # use the right ridge regression type
                if "ridge" == reg_type:
                    ridgereg = Ridge(alpha = alpha)
                elif "lasso" == reg_type:
                    ridgereg = linear_model.Lasso(alpha=alpha)
                else:
                    ridgereg = linear_model.ElasticNet(alpha=alpha)
                # set the polynomial degree
                poly = PolynomialFeatures(degree = pol_order)
                
                X_ = poly.fit_transform(X_train)
                ridgereg.fit(X_, Y_train)   
                
                X_ = poly.fit_transform(X_test)
                result = ridgereg.predict(X_)
                # calulcate the MSE
                score = metrics.mean_squared_error(result, Y_test)
                # write to csv data file
                writer.writerow([sys.argv[1], "PR - " + reg_type, airport_code, "N/A", pol_order, alpha, score, np.sqrt(score)])
                # save best score
                if score < score_min:
                    score_min = score
                    parameters = [alpha/10, pol_order]

                print("n={} alpha={} , MSE = {:<0.5}".format(pol_order, alpha, score))
                # print the average minutes our prediction was off by
                print('Ecart = {:.2f} min'.format(np.sqrt(score)))
        print("Best Score: " + str(score_min) + " Parameters: " + str(parameters))

   
def random_forest(tree_options = [10]):
    '''
    Random forest regression with a given list of tree sizes or a default size of 10
    '''
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for trees in tree_options:
            regressor = RandomForestRegressor(n_estimators=trees, random_state=False, criterion="mse")
            regressor.fit(X_train, Y_train)
            result = regressor.predict(X_test)
            # calculate MSE
            score = metrics.mean_squared_error(result, Y_test)
            # save the 10 tree model
            if trees == 10:
                rf_10 = regressor

            print("Score for %d trees: %f\n" % (trees, score))
            # write to csv
            writer.writerow([sys.argv[1], "RF", airport_code, trees, "N/A", "N/A", score, np.sqrt(score)])

    # dump the 10 tree random forest model into a file that we can load into the gui
    dump(rf_10, "rf_model.joblib")


if __name__ == "__main__":
    concatenate_data(int(sys.argv[1]))
    preprocess()
    random_forest([1, 5, 10, 20, 100])
    poly_regression("ridge")
    poly_regression("lasso")
    poly_regression("elastic")