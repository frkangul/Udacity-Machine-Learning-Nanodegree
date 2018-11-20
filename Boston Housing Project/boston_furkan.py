# Import libraries necessary for this project
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.cross_validation import ShuffleSplit
# Load the Boston housing dataset
# 'RM' is the average number of rooms among homes in the neighborhood.
# 'LSTAT' is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# 'PTRATIO' is the ratio of students to teachers in primary and secondary schools in the neighborhood.
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)   
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

# Data Exploration
# Minimum price of the data
minimum_price = np.min(prices)
# Maximum price of the data
maximum_price = np.max(prices)
# Mean price of the data
mean_price = np.mean(prices)
# Median price of the data
median_price = np.median(prices)
# Standard deviation of prices of the data
std_price = np.std(prices)
# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))
# Boxplot of prices to get a sense of the data
import matplotlib.pyplot as plt
#%matplotlib inline
plt.title("Boston Home Prices")
plt.ylabel("Price (USD)")
plt.boxplot(prices)
plt.show()

# Define a Performance Metric
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """  
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    # Return the score
    return score

# Shuffle and Split Data to remove any bias in the ordering of the dataset
from sklearn.cross_validation import train_test_split
# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)
# Success
print("Training and testing split was successful.")
print("Train shapes (X,y): ", X_train.shape, y_train.shape)
print("Test shapes (X,y): ", X_test.shape, y_test.shape)

# Analyzing Model Performance
vs.ModelLearning(features, prices)
vs.ModelComplexity(X_train, y_train)

# Fitting a Model
# Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()
    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':np.arange(1,11)}
    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)
    # Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)
# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

# Making Predictions
# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3
client_prices = []
# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
    client_prices.append(price)
    
# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))
print("Stds away from the mean (Client 1): ", (client_prices[0]-mean_price)/std_price)
print("Stds away from the mean (Client 2): ", (client_prices[1]-mean_price)/std_price)
print("Stds away from the mean (Client 3): ", (client_prices[2]-mean_price)/std_price)

# Sensitivity
vs.PredictTrials(features, prices, fit_model, client_data)