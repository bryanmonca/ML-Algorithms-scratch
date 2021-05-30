import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Moncada, Bryan

Summary:

Results using scikit-learn Perceptron model
Training set mean accuracy: 0.8289
Validation set mean accuracy: 0.7778
Testing set mean accuracy: 0.8200
Results using our Perceptron model trained with 10 steps
Training set mean accuracy: 0.5351
Validation set mean accuracy: 0.5714
Results using our Perceptron model trained with 20 steps
Training set mean accuracy: 0.7500
Validation set mean accuracy: 0.7619
Results using our Perceptron model trained with 60 steps
Training set mean accuracy: 0.8355
Validation set mean accuracy: 0.8889
Using best model trained with 60 steps
Testing set mean accuracy: 0.8400
'''

'''
Implementation of Perceptron for binary classification
'''
class PerceptronBinary(object):

    def __init__(self):
        # Define private variables
        self.__weights = None

    def __update(self, x, y):
        '''
        Update the weight vector during each training iteration

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # TODO: Implement the member update function

        # What is the dimension of x? (d, N)
        # Augment the feature vector x (d, N) with threshold
        threshold = 0.5 % np.ones([1, x.shape[1]])  # (1, N)
        x = np.concatenate([threshold, x], axis=0)  # (d+1, N)

        # Do the update for all incorrect predictions
        # w^(t+1) = w^(t) + y^n x^n

        # Walk through every example and check if they are incorrect
        for n in range(x.shape[1]):
            # x is (d+1, N), so the shape is (d+1), weights (d+1, 1)
            x_n = np.expand_dims(x[:, n], axis=-1)   # (d+1, 1)
            #y_n = np.expand_dims(y[:, n], axis=-1)
            # Predict the label for x_n
            prediction = np.sign(np.matmul(self.__weights.T, x_n))

            # Check if prediction is equal or not equal to ground truth y
            if (prediction != y[n]):
                # w^(t+1) = w^(t) + (y_n * x_n)
                # shape: (d+1) = (d+1, 1) + (1) * (d+1, 1)
                self.__weights = self.__weights + (y[n] * x_n)

    def fit(self, x, y, T=100, tol=1e-3):
        '''
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            t : int
                number of iterations to optimize perceptron
            tol : float
                change of loss tolerance, if greater than loss + tolerance, then stop
        '''
        # TODO: Implement the fit function
        self.__weights = np.zeros([x.shape[0] + 1, 1]) # (d+1, 1)
        self.__weights[0, 0] = -1.0

        # Initialize some previous loss and previous weights
        prev_loss = 2.0 # whatever
        prev_weights = np.copy(self.__weights) # what we started with
        for t in range(T):

            # Compute our loss
            predictions = self.predict(x)

            # l = 1/N \sum_n^N I(h(x^n) != y^n)
            loss = np.mean(np.where(predictions != y, 1.0, 0.0))
            print('t={} loss={}'.format(t + 1, loss))

            # Stopping conditions
            if (loss == 0.0):
                break
            elif (loss > prev_loss + tol and t > 2): # experience t>2, let's at least try for 3 steps
            # if our loss from t = 0.1, t+1 = 0.5, looks like we did worse
            # and should take weights of previous time step
            # take the weights of the previous time
                self.__weights = prev_weights
                break

            # Update previous loss and previous weights
            prev_loss = loss
            prev_weights = np.copy(self.__weights)

            # Updates our weight vector based on what we got wrong
            self.__update(x, y)

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                N x d feature vector

        Returns:
            numpy : 1 x N label vector
        '''
        # TODO: Implement the predict function

        # [w0, w1, w2, w3, w4, ..., wd] (d+1, N)
        # [  , x1, x2, x3, x4, ..., xd] (d, N)
        # What is the shape of threshold? (1 x N)
        threshold = 0.5 * np.ones([1, x.shape[1]])

        # Augment the features x with threshold
        x = np.concatenate((threshold, x), axis = 0)

        # Predictions using w^Tx: (d+1, 1)^T times (d+1, N) = (1, N)
        predictions = np.matmul(self.__weights.T, x) # (1, N)

        # What we are about is the sign of our predictions +/-
        return np.sign(predictions)

    def score(self, x, y):
        '''
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label

        Returns:
            float : mean accuracy
        '''
        # TODO: Implement the score function
        predictions = self.predict(x) # (1, N) of -1, +1

        # Comparing if predictions and y are the same
        scores = np.where(predictions == y, 1, 0)

        return np.mean(scores)




if __name__ == '__main__':

    breast_cancer_data = skdata.load_breast_cancer()
    x = breast_cancer_data.data
    y = breast_cancer_data.target

    # 80 percent train, 10 percent validation, 10 percent test split
    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % 9 == 0:
            val_idx.append(idx)
        elif idx and idx % 10 == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    '''
    Trains and tests Perceptron model from scikit-learn
    '''
    model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
    # Trains scikit-learn Perceptron model
    model.fit(x_train, y_train)

    print('Results using scikit-learn Perceptron model')

    # Test model on training set
    scores_train = model.score(x_train, y_train)
    print('Training set mean accuracy: {:.4f}'.format(scores_train))

    # Test model on validation set
    scores_val = model.score(x_val, y_val)
    print('Validation set mean accuracy: {:.4f}'.format(scores_val))

    # Test model on testing set
    scores_test = model.score(x_test, y_test)
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))

    '''
    Trains and tests our Perceptron model for binary classification
    '''
    # TODO: obtain dataset in correct shape (d x N) previously (Nxd)
    x_train = np.transpose(x_train, axes=(1,0))
    x_val = np.transpose(x_val, axes=(1,0))
    x_test = np.transpose(x_test, axes=(1,0))

    # TODO: obtain labels in {+1, -1} format
    y_train = np.where(y_train == 0, -1, 1)
    y_val = np.where(y_val == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # y shape from (N,) to (1xN)
    #y_train = np.expand_dims(y_train, -1).T
    #y_val = np.expand_dims(y_val, -1).T
    #y_test = np.expand_dims(y_test, -1).T

    # TODO: Initialize model, train model, score model on train, val and test sets

    # Train 3 PerceptronBinary models using 10, 20, and 60 steps with tolerance of 1
    models = []
    scores = []
    steps = [10, 20, 60]
    for T in steps:
        # Initialize PerceptronBinary model
        model = PerceptronBinary()
        models.append(model)

        print('Results using our Perceptron model trained with {} steps'.format(T))
        # Train model on training set
        model.fit(x_train, y_train, T=T, tol=1)

        # Test model on training set
        scores_train = model.score(x_train, y_train)
        print('Training set mean accuracy: {:.4f}'.format(scores_train))

        # Test model on validation set
        scores_val = model.score(x_val, y_val)
        scores.append(scores_val)
        print('Validation set mean accuracy: {:.4f}'.format(scores_val))

        # Save the model and its score

    # Select the best performing model on the validation set
    best_idx = np.argmax(scores)

    print('Using best model trained with {} steps'.format(steps[best_idx]))

    # Test model on testing set
    scores_test = models[best_idx].score(x_test, y_test)
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))
