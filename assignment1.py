import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Moncada, Bryan

Summary:

1) What did you do in this assignment?
In this assignment, it was implemented the Multi-Class Perceptron Algorithm for classification.
Then, We used the implementation with two datasets: Iris and Wine, and compare the results of the 
sklearn Perceptron class with the results of our implementation. In theory, the results should
be very similar. To get closer to sklearn results, an important step was to tune the hyper-parameters
of our model.

2) How did you do it?
The implementation of this Algorithm is very similar to the Perceptron Learning Algorithm for only
two classes. There are two important things that are different in the Multi-Class Perceptron:
First, because we now have `c` classes, the hypothesis is going to be the `arg max c (w_c.T x)`. 
This means that we choose the best class based on which class give us the highest score 
in `w.T x`.
Second, the way we update the weights is also different. Now, we update two set of weights every
time we have a wrong prediction, the weights of the class of the hypothesis we chose and the weights 
of the class with the true label.
Also, to get to the best result, it is important to change the hyper-parameteres and see how our loss
is evolving. The number of steps was changed depending on on which step the loss reached a minimum.

3) What are the constants and hyper-parameters you used?
constants used:
    threshold = 0.5
    w0 = -1
    initial loss = 2.0
final hyper-parameters:
    For Iris dataset:
        training steps = [15, 38, 60]
        tolerance = 1.0
    For Wine dataset:
        training steps = [24, 38, 75]
        tolerance = 1.0
Finally, the results obtained were very similar to sklearn's results

Scores:

Results on the iris dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.8512
Validation set mean accuracy: 0.7333
Testing set mean accuracy: 0.9286
Results on the iris dataset using our Perceptron model trained with 15 steps and tolerance of 1.0
Training set mean accuracy: 0.9752
Validation set mean accuracy: 1.0000
Results on the iris dataset using our Perceptron model trained with 38 steps and tolerance of 1.0
Training set mean accuracy: 0.9587
Validation set mean accuracy: 1.0000
Results on the iris dataset using our Perceptron model trained with 60 steps and tolerance of 1.0
Training set mean accuracy: 0.9752
Validation set mean accuracy: 1.0000
Using best model trained with 15 steps and tolerance of 1.0
Testing set mean accuracy: 0.9286
Results on the wine dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.5625
Validation set mean accuracy: 0.4118
Testing set mean accuracy: 0.4706
Results on the wine dataset using our Perceptron model trained with 24 steps and tolerance of 1.0
Training set mean accuracy: 0.4167
Validation set mean accuracy: 0.4706
Results on the wine dataset using our Perceptron model trained with 38 steps and tolerance of 1.0
Training set mean accuracy: 0.4444
Validation set mean accuracy: 0.4118
Results on the wine dataset using our Perceptron model trained with 75 steps and tolerance of 1.0
Training set mean accuracy: 0.5694
Validation set mean accuracy: 0.4706
Using best model trained with 24 steps and tolerance of 1.0
Testing set mean accuracy: 0.4706
'''

'''
Implementation of Perceptron for multi-class classification
'''
class PerceptronMultiClass(object):

    def __init__(self):
        # Define private variables, weights and number of classes
        self.__weights = None
        self.__n_class = -1

    def __update(self, x, y):
        '''
        Update the weight vector during each training iteration

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
        '''
        predictions = self.predict(x) # we can reuse or predict function, in there x will become (d+1, N) 

        threshold = 0.5 * np.ones([1, x.shape[1]])  # however, we need to expand x dim in this function 
        x = np.concatenate((threshold, x), axis = 0)

        for n in range(x.shape[1]):
            predicition = predictions[n]
            if (predicition != y[n]):                
                c_hat = predicition
                c_star = y[n]
                self.__weights[:, c_hat] = self.__weights[:, c_hat] - x[:, n]
                self.__weights[:, c_star] = self.__weights[:, c_star] + x[:, n]

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
        self.__n_class = np.unique(y).shape[0]
        self.__weights = np.zeros([x.shape[0] + 1, self.__n_class]) # (d+1, n)
        self.__weights[0, :] = -1.0

        prev_loss = 2.0
        prev_weights = np.copy(self.__weights)
        for t in range(T):
            predictions = self.predict(x)
            loss = np.mean(np.where(predictions != y, 1.0, 0.0))
            #print('t={} loss={}'.format(t + 1, loss))

            if (loss == 0.0):
                break
            elif (loss > prev_loss + tol and t > 2):
                self.__weights = prev_weights
                break
            
            prev_loss = loss
            prev_weights = np.copy(self.__weights)

            self.__update(x, y)


    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : 1 x N label vector
        '''
        threshold = 0.5 * np.ones([1, x.shape[1]])
        x = np.concatenate((threshold, x), axis = 0)

        predictions = np.zeros([self.__n_class, x.shape[1]]) # initialize predictions for each class (c, N)
        for c in range(self.__n_class):
            weights_c = np.expand_dims(self.__weights[:, c], axis=-1) # (d+1, 1)
            predictions_c = np.matmul(weights_c.T, x) # (1, N)
            predictions[c, :] = predictions_c
        
        return np.argmax(predictions, axis=0)  # argmax of each predicition

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
        predictions = self.predict(x) # (1 x N)
        scores = np.where(predictions == y, 1, 0)

        return np.mean(scores)


def split_dataset(x, y, n_sample_train_to_val_test=8):
    '''
    Helper function to splits dataset into training, validation and testing sets

    Args:
        x : numpy
            d x N feature vector
        y : numpy
            1 x N ground-truth label
        n_sample_train_to_val_test : int
            number of training samples for every validation, testing sample

    Returns:
        x_train : numpy
            d x n feature vector
        y_train : numpy
            1 x n ground-truth label
        x_val : numpy
            d x m feature vector
        y_val : numpy
            1 x m ground-truth label
        x_test : numpy
            d x m feature vector
        y_test : numpy
            1 x m ground-truth label
    '''
    n_sample_interval = n_sample_train_to_val_test + 2

    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % n_sample_interval == (n_sample_interval - 1):
            val_idx.append(idx)
        elif idx and idx % n_sample_interval == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':

    iris_data = skdata.load_iris()
    wine_data = skdata.load_wine()

    datasets = [iris_data, wine_data]
    tags = ['iris', 'wine']

    # Experiment with 3 different max training steps (T) for each dataset
    train_steps_iris = [15, 38, 60]     # 28
    train_steps_wine = [24, 38, 75]    # 48 #63 

    train_steps = [train_steps_iris, train_steps_wine]

    # Set a tolerance for each dataset
    tol_iris = 1
    tol_wine = 1

    tols = [tol_iris, tol_wine]

    for dataset, steps, tol, tag in zip(datasets, train_steps, tols, tags):
        # Split dataset into 80 training, 10 validation, 10 testing
        x = dataset.data
        y = dataset.target
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(
            x=x,
            y=y,
            n_sample_train_to_val_test=8)

        '''
        Trains and tests Perceptron model from scikit-learn
        '''
        model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
        # Trains scikit-learn Perceptron model
        model.fit(x_train, y_train)

        print('Results on the {} dataset using scikit-learn Perceptron model'.format(tag))

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
        Trains, validates, and tests our Perceptron model for multi-class classification
        '''
        # Obtain dataset in correct shape (d x N)
        x_train = np.transpose(x_train, axes=(1,0))
        x_val = np.transpose(x_val, axes=(1,0))
        x_test = np.transpose(x_test, axes=(1,0))

        # Initialize empty lists to hold models and scores
        models = []
        scores = []
        for T in steps:
            # Initialize PerceptronMultiClass model
            model = PerceptronMultiClass()

            print('Results on the {} dataset using our Perceptron model trained with {} steps and tolerance of {}'.format(tag, T, tol))
            # Train model on training set
            model.fit(x_train, y_train, T=T, tol=tol)

            # Test model on training set
            scores_train = model.score(x_train, y_train)
            print('Training set mean accuracy: {:.4f}'.format(scores_train))

            # Test model on validation set
            scores_val = model.score(x_val, y_val)
            
            print('Validation set mean accuracy: {:.4f}'.format(scores_val))

            # Save the model and its score
            models.append(model)
            scores.append(scores_val)

        # Select the best performing model on the validation set
        best_idx = np.argmax(scores)

        print('Using best model trained with {} steps and tolerance of {}'.format(steps[best_idx], tol))

        # Test model on testing set
        scores_test = models[best_idx].score(x_test, y_test)
        print('Testing set mean accuracy: {:.4f}'.format(scores_test))
