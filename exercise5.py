import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from itertools import combinations_with_replacement


'''
Name: Moncada, Bryan

Summary:

Results using scikit-learn LinearRegression model with linear features
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results of LinearRegression model using scikit-learn order-2 polynomial expansion features
Training set mean squared error: 8.8948
Training set r-squared scores: 0.8976
Validation set mean squared error: 11.4985
Validation set r-squared scores: 0.8360
Testing set mean squared error: 34.8401
Testing set r-squared scores: 0.5539
Results of LinearRegression model using scikit-learn order-3 polynomial expansion features
Training set mean squared error: 0.0000
Training set r-squared scores: 1.0000
Validation set mean squared error: 131227.1239
Validation set r-squared scores: -1870.9951
Testing set mean squared error: 119705.3692
Testing set r-squared scores: -1531.7148
Results for LinearRegression model using our implementation of order-2 polynomial expansion features
Training set mean squared error: 8.8948
Training set r-squared scores: 0.8976
Validation set mean squared error: 11.4985
Validation set r-squared scores: 0.8360
Testing set mean squared error: 34.8401
Testing set r-squared scores: 0.5539
Results for LinearRegression model using our implementation of order-3 polynomial expansion features
Training set mean squared error: 0.0000
Training set r-squared scores: 1.0000
Validation set mean squared error: 131226.8829
Validation set r-squared scores: -1870.9916
Testing set mean squared error: 119705.0318
Testing set r-squared scores: -1531.7105
'''

'''
Implementation of our polynomial expansion for nonlinear mapping
'''
class PolynomialFeatureExpansion(object):

    def __init__(self, degree):
        '''
        Args:
            degree : int
                order or degree of polynomial for expansion
        '''
        # Degree or order of polynomial we will expand to
        self.__degree = degree

    def transform(self, X):
        '''
        Computes up to p-order (degree) polynomial features and augments them to the data by
        1. We need to add a bias (x_0)
        2. We need to augment up to p-order polynomial

        Args:
            X : numpy
                N x d feature vector

        Returns:
            polynomial expanded features in Z space
        '''

        # Initialize the bias
        bias = np.ones([X.shape[0], 1])

        # Initialize polynomial expansion features Z
        # Z contains [x_0, x_1, x_2, ..., x_d]
        Z = [bias, X]

        # If degree is less than 2, then return the original features
        if self.__degree < 2:
            Z = np.concatenate(Z, axis=1)
            return Z

        # Split X into it's d dimensions separately
        linear_features = np.split(X, indices_or_sections=X.shape[1], axis=1)

        indexes_list = list(range(len(linear_features)))

        list_index_combinations = []  # this will collect index combinations for 2nd, 3rd, 4th .... D order
        for i in range(2, self.__degree+1):
            index_combinations = list(combinations_with_replacement(indexes_list, i))   # list of tuples of unique index combinations
            list_index_combinations.append(index_combinations)
        
        new_polynomial_features = []
        for index_combinations in list_index_combinations:       
            for tup in index_combinations:
                polynomial_feature = np.ones([X.shape[0], 1])
                for i in tup:
                    polynomial_feature = polynomial_feature * linear_features[i] 
                new_polynomial_features.append(polynomial_feature)
        Z.append(np.concatenate(new_polynomial_features, axis=1))
            
        
        #Z.append(np.concatenate(new_polynomial_features, axis=1))

        # Concatenate every term into the feature vector (augmenting X with polynomial features)
        # Z becomes [x_0, x_1, x_2, x_1^2, x_1 x_2, x_2^2, x_1^3, x_1^2 x_2, x_1 x_2^2, x_2^3] of shape (N, 10)
        Z = np.concatenate(Z, axis=1)

        return Z


if __name__ == '__main__':

    boston_housing_data = skdata.load_boston()
    x = boston_housing_data.data
    y = boston_housing_data.target

    # 80 percent train, 10 percent validation, 10 percent test split
    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % 10 == 9:
            val_idx.append(idx)
        elif idx and idx % 10 == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    '''
    Trains and tests linear regression from scikit-learn
    '''
    # Variables to hold training, validation and testing scores for linear up to p-order polynomial expansion
    scores_mse_train = []
    scores_mse_val = []
    scores_mse_test = []
    scores_r2_train = []
    scores_r2_val = []
    scores_r2_test = []

    # Initialize scikit-learn linear regression model
    model = LinearRegression()

    # Trains scikit-learn linear regression model
    model.fit(x_train, y_train)

    print('Results using scikit-learn LinearRegression model with linear features')

    # Test model on training set
    predictions_train = model.predict(x_train)

    score_mse_train = skmetrics.mean_squared_error(predictions_train, y_train)
    print('Training set mean squared error: {:.4f}'.format(score_mse_train))

    score_r2_train = model.score(x_train, y_train)
    print('Training set r-squared scores: {:.4f}'.format(score_r2_train))

    # TODO: Save MSE and R-square scores on training set
    scores_mse_train.append(score_mse_train)
    scores_r2_train.append(score_r2_train)

    # Test model on validation set
    predictions_val = model.predict(x_val)

    score_mse_val = skmetrics.mean_squared_error(predictions_val, y_val)
    print('Validation set mean squared error: {:.4f}'.format(score_mse_val))

    score_r2_val = model.score(x_val, y_val)
    print('Validation set r-squared scores: {:.4f}'.format(score_r2_val))

    # TODO: Save MSE and R-square scores on validation set
    scores_mse_val.append(score_mse_val)
    scores_r2_val.append(score_r2_val)

    # Test model on testing set
    predictions_test = model.predict(x_test)

    score_mse_test = skmetrics.mean_squared_error(predictions_test, y_test)
    print('Testing set mean squared error: {:.4f}'.format(score_mse_test))

    score_r2_test = model.score(x_test, y_test)
    print('Testing set r-squared scores: {:.4f}'.format(score_r2_test))

    # TODO: Save MSE and R-square scores on testing set
    scores_mse_test.append(score_mse_test)
    scores_r2_test.append(score_r2_test)

    # TODO: Set the degrees/orders of polynomials to be 2 and 3 for nonlinear mapping
    degrees_polynomial = [2, 3, 4]

    '''
    Trains and tests linear regression from scikit-learn with scikit-learn polynomial features
    '''
    for degree in degrees_polynomial:

        print('Results of LinearRegression model using scikit-learn order-{} polynomial expansion features'.format(degree))

        # TODO: Initialize polynomial expansion
        poly_transform = skpreprocess.PolynomialFeatures(degree=degree)

        # TODO: Compute the polynomial terms needed for the data
        # Generates x_1^2, x_1 x_2, x_1 x_3, ..., x_d^2
        poly_transform.fit(x_train)

        # TODO: Transform the data by nonlinear mapping
        # Applies all the polynomial terms to the data and augments it to x
        # Computes the values for x_0, x_1, x_2, ..., x_1^2, x_1 x_2, ... x_d^2
        # x_1 = 2, x_2 = 4 : x -> (1, 2, 4, ..., 4, 8, ..., x_d^2)
        x_poly_train = poly_transform.transform(x_train)
        #print(x_poly_train.shape)
        x_poly_val = poly_transform.transform(x_val)
        x_poly_test = poly_transform.transform(x_test)

        # Initialize scikit-learn linear regression model
        model_poly = LinearRegression()

        # TODO: Trains scikit-learn linear regression model using p-order polynomial expansion
        model_poly.fit(x_poly_train, y_train)

        # TODO: Test model on training set
        predictions_poly_train = model_poly.predict(x_poly_train)

        score_mse_poly_train = skmetrics.mean_squared_error(predictions_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_poly_train))

        score_r2_poly_train = model_poly.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_poly_train))

        # TODO: Save MSE and R-square scores on training set
        scores_mse_train.append(score_mse_poly_train)
        scores_r2_train.append(score_r2_poly_train)

        # TODO: Test model on validation set
        predictions_poly_val = model_poly.predict(x_poly_val)

        score_mse_poly_val = skmetrics.mean_squared_error(predictions_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_poly_val))

        score_r2_poly_val = model_poly.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_poly_val))

        # TODO: Save MSE and R-square scores on validation set
        scores_mse_val.append(score_mse_poly_val)
        scores_r2_val.append(score_r2_poly_val)

        # TODO: Test model on testing set
        predictions_poly_test = model_poly.predict(x_poly_test)

        score_mse_poly_test = skmetrics.mean_squared_error(predictions_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_poly_test))

        score_r2_poly_test = model_poly.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_poly_test))

        # TODO: Save MSE and R-square scores on testing set
        scores_mse_test.append(score_mse_poly_test)
        scores_r2_test.append(score_r2_poly_test)

    # TODO: Convert each scores to NumPy arrays
    scores_mse_train = np.array(scores_mse_train)
    scores_mse_val = np.array(scores_mse_val)
    scores_mse_test = np.array(scores_mse_test)
    scores_r2_train = np.array(scores_r2_train)
    scores_r2_val = np.array(scores_r2_val)
    scores_r2_test = np.array(scores_r2_test)

    # TODO: Clip each set of MSE scores between 0 and 50
    scores_mse_train = np.clip(scores_mse_train, 0.0, 50.0)
    scores_mse_val = np.clip(scores_mse_val, 0.0, 50.0)
    scores_mse_test = np.clip(scores_mse_test, 0.0, 50.0)

    # TODO: Clip each set of R-squared scores between 0 and 1
    scores_r2_train = np.clip(scores_r2_train, 0.0, 1.0)
    scores_r2_val = np.clip(scores_r2_val, 0.0, 1.0)
    scores_r2_test = np.clip(scores_r2_test, 0.0, 1.0)

    n_experiments = len(scores_mse_train)

    # TODO: Create figure for training, validation and testing scores for different features
    fig = plt.figure()

    # TODO: Create subplot for MSE for training, validation, testing
    ax1 = fig.add_subplot(1, 2, 1)

    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_mse_train, scores_mse_val, scores_mse_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Plot MSE scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax1.plot(x, y, marker='o', color=color, label=label)
        ax1.legend(loc='best')

    # TODO: Set y limits between 0 and 50, set x limits to 0 to number experiments + 1
    ax1.set_ylim([0.0, 50.0])
    ax1.set_xlim([0.0, n_experiments + 1])

    # TODO: Set y label to 'MSE', set x label to 'p-degree'
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('p-degree')

    # TODO: Create subplot for R-square for training, validation, testing
    ax2 = fig.add_subplot(1, 2, 2)

    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_r2_train, scores_r2_val, scores_r2_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Plot R-squared scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax2.plot(x, y, marker='o', color=color, label=label)
        ax2.legend(loc='best')

    # TODO: Set y limits between 0 and 1, set x limits to 0 to number experiments + 1
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlim([0.0, n_experiments + 1])

    # TODO: Set y label to 'R-squared', set x label to 'p-degree'
    ax2.set_ylabel('R-squared')
    ax2.set_xlabel('p-degree')

    # TODO: Create super title 'Scikit-learn Polynomial Expansion on Training, Validation and Testing Sets'
    plt.suptitle('Scikit-learn Polynomial Expansion on Training, Validation and Testing Sets')

    '''
    Trains and tests linear regression from scikit-learn with our implementation of polynomial features
    '''
    # TODO: Instantiate lists containing the training, validation and testing
    # MSE and R-squared scores obtained from linear regression without nonlinear mapping
    scores_mse_train = [scores_mse_train[0]]
    scores_mse_val = [scores_mse_val[0]]
    scores_mse_test = [scores_mse_test[0]]
    scores_r2_train = [scores_r2_train[0]]
    scores_r2_val = [scores_r2_val[0]]
    scores_r2_test = [scores_r2_test[0]]
    
    for degree in degrees_polynomial:

        print('Results for LinearRegression model using our implementation of order-{} polynomial expansion features'.format(degree))

        # TODO: Transform the data by nonlinear mapping using our implementation of polynomial expansion
        poly_transform = PolynomialFeatureExpansion(degree)

        x_poly_train = poly_transform.transform(x_train)
        #print(x_poly_train.shape)
        x_poly_val = poly_transform.transform(x_val)
        x_poly_test = poly_transform.transform(x_test)        

        # Initialize scikit-learn linear regression model
        model_poly = LinearRegression()

        # TODO: Trains scikit-learn linear regression model using p-order polynomial expansion
        model_poly.fit(x_poly_train, y_train)

        # TODO: Test model on training set
        predictions_poly_train = model_poly.predict(x_poly_train)

        score_mse_poly_train = skmetrics.mean_squared_error(predictions_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_poly_train))

        score_r2_poly_train = model_poly.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_poly_train))

        # TODO: Save MSE and R-square scores on training set
        scores_mse_train.append(score_mse_poly_train)
        scores_r2_train.append(score_r2_poly_train)        

        # TODO: Test model on validation set
        predictions_poly_val = model_poly.predict(x_poly_val)

        score_mse_poly_val = skmetrics.mean_squared_error(predictions_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_poly_val))

        score_r2_poly_val = model_poly.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_poly_val))

        # TODO: Save MSE and R-square scores on validation set
        scores_mse_val.append(score_mse_poly_val)
        scores_r2_val.append(score_r2_poly_val)

        # TODO: Test model on testing set
        predictions_poly_test = model_poly.predict(x_poly_test)

        score_mse_poly_test = skmetrics.mean_squared_error(predictions_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_poly_test))

        score_r2_poly_test = model_poly.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_poly_test))

        # TODO: Save MSE and R-square scores on testing set
        scores_mse_test.append(score_mse_poly_test)
        scores_r2_test.append(score_r2_poly_test)

    # TODO: Convert each scores to NumPy arrays
    scores_mse_train = np.array(scores_mse_train)
    scores_mse_val = np.array(scores_mse_val)
    scores_mse_test = np.array(scores_mse_test)
    scores_r2_train = np.array(scores_r2_train)
    scores_r2_val = np.array(scores_r2_val)
    scores_r2_test = np.array(scores_r2_test)

    # TODO: Clip each set of MSE scores between 0 and 50
    scores_mse_train = np.clip(scores_mse_train, 0.0, 50.0)
    scores_mse_val = np.clip(scores_mse_val, 0.0, 50.0)
    scores_mse_test = np.clip(scores_mse_test, 0.0, 50.0)

    # TODO: Clip each set of R-squared scores between 0 and 1
    scores_r2_train = np.clip(scores_r2_train, 0.0, 1.0)
    scores_r2_val = np.clip(scores_r2_val, 0.0, 1.0)
    scores_r2_test = np.clip(scores_r2_test, 0.0, 1.0)

    n_experiments = len(scores_mse_train)

    # TODO: Create figure for training, validation and testing scores for different features
    fig = plt.figure()

    # TODO: Create subplot for MSE for training, validation, testing
    ax1 = fig.add_subplot(1, 2, 1)

    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_mse_train, scores_mse_val, scores_mse_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Plot MSE scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax1.plot(x, y, marker='o', color=color, label=label)
        ax1.legend(loc='best')

    # TODO: Set y limits between 0 and 50, set x limits to 0 to number experiments + 1
    ax1.set_ylim([0.0, 50.0])
    ax1.set_xlim([0.0, n_experiments + 1])

    # TODO: Set y label to 'MSE', set x label to 'p-degree'
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('p-degree')

    # TODO: Create subplot for R-square for training, validation, testing
    ax2 = fig.add_subplot(1, 2, 2)

    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_r2_train, scores_r2_val, scores_r2_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Plot R-squared scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax2.plot(x, y, marker='o', color=color, label=label)
        ax2.legend(loc='best')

    # TODO: Set y limits between 0 and 1, set x limits to 0 to number experiments + 1
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlim([0.0, n_experiments + 1])

    # TODO: Set y label to 'R-squared', set x label to 'p-degree'
    ax2.set_ylabel('R-squared')
    ax2.set_xlabel('p-degree')

    # TODO: Create super title 'Our Polynomial Expansion on Training, Validation and Testing Sets'
    plt.suptitle('Our Polynomial Expansion on Training, Validation and Testing Sets')

    plt.show()
