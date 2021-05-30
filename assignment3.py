import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression


'''
Name: Moncada, Bryan

Summary:

1. What did you observe when using larger versus smaller momentum for
momentum gradient descent and momentum stochastic gradient descent?

Momentum helped us speed convergence because we are considering a fraction
of the previous gradients. In addition, with SGD, momentum reduced a little
bit the variance introduced by stochastic sampling. 

2. What did you observe when using larger versus smaller batch size
for stochastic gradient descent?

Variance depends on the size of the batch. As the batch decreases, the variance
gets bigger. High variance takes a longer time to converge. In the implementation,
a smaller batch size with the same number of steps resulted in a higher loss.

3. Explain the difference between gradient descent, momentum gradient descent,
stochastic gradient descent, and momentum stochastic gradient descent?

Gradient descent is a local approximation of the full function or the full loss
at some local point in time. In the exercise, we only used around 6000 time steps
and a learning rate 0f 0.4. 
However, we are not considering the previous gradients. Using momentum with 
gradient descent helps us speed up convergence. We could see that when printing
the losses.
Stochastic Gradient Descent (SGD) approximates the approximation of Gradient Descent
using a batch of the data. Because we do not have the full gradient, we will
have a higher variance and it is harder to converge. More time steps are needed. 
In the implementation we used 20,000.
Similar as Momentum gradient descent, Momentum SGD considers a moving average of the
previous gradients. When doing random sampling, a lot of noise could be inserted,
momentum helps us reduce the amount of noise. Now, the direction that we take 
can be more "correct". In the exercise, we could get a better result with
momentum SGD using the same parameters as SGD.


Report your scores here.

Results on using scikit-learn Ridge Regression model
Training set mean squared error: 2749.2155
Validation set mean squared error: 3722.5782
Testing set mean squared error: 3169.6860
Results on using Ridge Regression using gradient descent variants
Fitting with gradient_descent using learning rate=4.0E-01, t=6000
Training set mean squared error: 2755.7929
Validation set mean squared error: 3729.6440
Testing set mean squared error: 3170.5261
Fitting with momentum_gradient_descent using learning rate=5.0E-01, t=6000
Training set mean squared error: 2753.7337
Validation set mean squared error: 3727.5260
Testing set mean squared error: 3170.4822
Fitting with stochastic_gradient_descent using learning rate=1.5E+00, t=20000
Training set mean squared error: 2780.5460
Validation set mean squared error: 3742.6051
Testing set mean squared error: 3194.8308
Fitting with momentum_stochastic_gradient_descent using learning rate=1.5E+00, t=20000
Training set mean squared error: 2783.9318
Validation set mean squared error: 3757.4962
Testing set mean squared error: 3157.8819
'''


def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric

    Args:
        model : object
            trained model, assumes predict function returns N x d predictions
        x : numpy
            d x N numpy array of features
        y : numpy
            N element groundtruth vector
    Returns:
        float : mean squared error
    '''

    # Computes the mean squared error
    predictions = model.predict(x)
    mse = skmetrics.mean_squared_error(predictions, y)
    return mse


'''
Implementation of our gradient descent optimizer for ridge regression
'''
class GradientDescentOptimizer(object):

    def __init__(self, learning_rate):
        self.__momentum = None
        self.__learning_rate = learning_rate

    def __compute_gradients(self, w, x, y, lambda_weight_decay):
        '''
        Returns the gradient of the mean squared loss with weight decay

        Args:
            w : numpy
                d x 1 weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            lambda_weight_decay : float
                weight of weight decay

        Returns:
            numpy : 1 x d gradients
        '''

        # TODO: Implements the __compute_gradients function 
        
        # Add bias to x (d x N) -> (d+1, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        fidelity_grad = (2.0/x.shape[1]) * np.matmul((np.matmul(x.T, w) - y), x.T)
        reg_grad = (2.0/x.shape[1]) * lambda_weight_decay * w
        gradients = fidelity_grad + reg_grad

        return gradients

    def __cube_root_decay(self, time_step):
        '''
        Computes the cube root polynomial decay factor t^{-1/3}

        Args:
            time_step : int
                current step in optimization

        Returns:
            float : cube root decay factor to adjust learning rate
        '''

        # TODO: Implement cube root polynomial decay factor to adjust learning rate

        return time_step ** (-1/3)

    def update(self,
               w,
               x,
               y,
               optimizer_type,
               lambda_weight_decay,
               beta,
               batch_size,
               time_step):
        '''
        Updates the weight vector based on

        Args:
            w : numpy
                1 x d weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            optimizer_type : str
                'gradient_descent',
                'momentum_gradient_descent',
                'stochastic_gradient_descent',
                'momentum_stochastic_gradient_descent'
            lambda_weight_decay : float
                weight of weight decay
            beta : str
                momentum discount rate
            batch_size : int
                batch size for stochastic and momentum stochastic gradient descent
            time_step : int
                current step in optimization

        Returns:
            numpy : 1 x d weights
        '''

        # TODO: Implement the optimizer update function

        if self.__momentum is None:
            self.__momentum = np.zeros_like(w)

        if optimizer_type == 'gradient_descent':

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)

            # TODO: Update weights
            return w - self.__learning_rate * gradients

        elif optimizer_type == 'momentum_gradient_descent':

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)

            # TODO: Compute momentum
            self.__momentum = beta * self.__momentum + (1 - beta) * gradients

            # TODO: Update weights
            return w - self.__learning_rate * self.__momentum

        elif optimizer_type == 'stochastic_gradient_descent':

            # TODO: Implement stochastic gradient descent

            # TODO: Sample batch from dataset
            b = np.random.randint(x.shape[1], size=batch_size)
            x_batch = x[:, b]
            y_batch = y[b]

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x_batch, y_batch, lambda_weight_decay)

            # TODO: Compute cube root decay factor and multiply by learning rate
            eta = self.__cube_root_decay(time_step) * self.__learning_rate

            # TODO: Update weights
            return w - eta * gradients

        elif optimizer_type == 'momentum_stochastic_gradient_descent':

            # TODO: Implement momentum stochastic gradient descent

            # TODO: Sample batch from dataset
            b = np.random.randint(x.shape[1], size=batch_size)
            x_batch = x[:, b]
            y_batch = y[b]

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x_batch, y_batch, lambda_weight_decay)

            # TODO: Compute momentum
            self.__momentum = beta * self.__momentum + (1 - beta) * gradients

            # TODO: Compute cube root decay factor and multiply by learning rate
            eta = self.__cube_root_decay(time_step) * self.__learning_rate

            # TODO: Update weights
            return w - eta * self.__momentum


'''
Implementation of our Ridge Regression model trained using gradient descent variants
'''
class RidgeRegressionGradientDescent(object):

    def __init__(self):
        # Define private variables
        self.__weights = None
        self.__optimizer = None

    def fit(self,
            x,
            y,
            optimizer_type,
            learning_rate,
            t,
            lambda_weight_decay,
            beta,
            batch_size):
        '''
        Fits the model to x and y by updating the weight vector
        using gradient descent variants

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            optimizer_type : str
                'gradient_descent',
                'momentum_gradient_descent',
                'stochastic_gradient_descent',
                'momentum_stochastic_gradient_descent'
            learning_rate : float
                learning rate
            t : int
                number of iterations to train
            lambda_weight_decay : float
                weight of weight decay
            beta : str
                momentum discount rate
            batch_size : int
                batch size for stochastic and momentum stochastic gradient descent
        '''

        # TODO: Implement the fit function

        # TODO: Initialize weights
        self.__weights = np.zeros([x.shape[0] + 1])
        self.__weights[0] = 1.0

        # TODO: Initialize optimizer
        self.__optimizer = GradientDescentOptimizer(learning_rate)

        for time_step in range(1, t + 1):

            # TODO: Compute loss function
            loss, loss_data_fidelity, loss_regularization = self.__compute_loss(x, y, lambda_weight_decay)

            if (time_step % 500) == 0:
                print('Step={:5}  Loss={:.4f}  Data Fidelity={:.4f}  Regularization={:.4f}'.format(
                    time_step, loss, loss_data_fidelity, loss_regularization))

            # TODO: Update weights
            self.__weights = self.__optimizer.update(
                       self.__weights,
                       x,
                       y,
                       optimizer_type,
                       lambda_weight_decay,
                       beta,
                       batch_size,
                       t)

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : N element vector
        '''

        # TODO: Implements the predict function

        # Add bias to x -> (d+1, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        return np.matmul(self.__weights.T, x)

    def __compute_loss(self, x, y, lambda_weight_decay):
        '''
        Returns the gradient of the mean squared loss with weight decay

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            lambda_weight_decay : float
                weight of weight decay

        Returns:
            float : loss
            float : loss data fidelity
            float : loss regularization
        '''

        # TODO: Implements the __compute_loss function
        loss_data_fidelity = np.mean((self.predict(x) - y)**2)
        loss_regularization = lambda_weight_decay * np.matmul(self.__weights.T, self.__weights) / (x.shape[1]+1)
        loss = loss_data_fidelity + loss_regularization

        return loss, loss_data_fidelity, loss_regularization


if __name__ == '__main__':

    # Loads dataset with 80% training, 10% validation, 10% testing split
    data = skdata.load_diabetes()
    x = data.data
    y = data.target

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

    x_train, x_val, x_test = \
        x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = \
        y[train_idx], y[val_idx], y[test_idx]

    # Initialize polynomial expansion

    poly_transform = skpreprocess.PolynomialFeatures(degree=2)

    # Compute the polynomial terms needed for the data
    poly_transform.fit(x_train)

    # Transform the data by nonlinear mapping
    x_train = poly_transform.transform(x_train)
    x_val = poly_transform.transform(x_val)
    x_test = poly_transform.transform(x_test)

    lambda_weight_decay = 0.1

    '''
    Trains and tests Ridge Regression model from scikit-learn
    '''

    # Trains scikit-learn Ridge Regression model on diabetes data
    ridge_scikit = RidgeRegression(alpha=lambda_weight_decay)
    ridge_scikit.fit(x_train, y_train)

    print('Results on using scikit-learn Ridge Regression model')

    # Test model on training set
    scores_mse_train_scikit = score_mean_squared_error(
        ridge_scikit, x_train, y_train)
    print('Training set mean squared error: {:.4f}'.format(scores_mse_train_scikit))

    # Test model on validation set
    scores_mse_val_scikit = score_mean_squared_error(
        ridge_scikit, x_val, y_val)
    print('Validation set mean squared error: {:.4f}'.format(scores_mse_val_scikit))

    # Test model on testing set
    scores_mse_test_scikit = score_mean_squared_error(
        ridge_scikit, x_test, y_test)
    print('Testing set mean squared error: {:.4f}'.format(scores_mse_test_scikit))

    '''
    Trains and tests our Ridge Regression model trained using gradient descent variants
    '''

    # Optimization types to use
    optimizer_types = [
        'gradient_descent',
        'momentum_gradient_descent',
        'stochastic_gradient_descent',
        'momentum_stochastic_gradient_descent'
    ]

    # TODO: Select learning rates for each optimizer
    learning_rates = [0.4, 0.5, 1.5, 1.5]

    # TODO: Select number of steps (t) to train
    T = [6000, 6000, 20000, 20000]

    # TODO: Select beta for momentum (do not replace None)
    betas = [None, 0.05, None, 0.25]

    # TODO: Select batch sizes for stochastic and momentum stochastic gradient descent (do not replace None)
    batch_sizes = [None, None, 256, 256]

    # TODO: Convert dataset (N x d) to correct shape (d x N)
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1,0))
    x_test = np.transpose(x_test, axes=(1,0))

    print('Results on using Ridge Regression using gradient descent variants')

    hyper_parameters = \
        zip(optimizer_types, learning_rates, T, betas, batch_sizes)

    for optimizer_type, learning_rate, t, beta, batch_size in hyper_parameters:

        # Conditions on batch size and beta
        if batch_size is not None:
            assert batch_size <= 0.90 * x_train.shape[1]

        if beta is not None:
            assert beta >= 0.05

        # TODO: Initialize ridge regression trained with gradient descent variants
        ridge_grad_descent = RidgeRegressionGradientDescent()

        print('Fitting with {} using learning rate={:.1E}, t={}'.format(
            optimizer_type, learning_rate, t))

        # TODO: Train ridge regression using gradient descent variants
        ridge_grad_descent.fit(
                x=x_train,
                y=y_train,
                optimizer_type=optimizer_type,
                learning_rate=learning_rate,
                t=t,
                lambda_weight_decay=lambda_weight_decay,
                beta=beta,
                batch_size=batch_size)

        # TODO: Test model on training set
        score_mse_grad_descent_train = score_mean_squared_error(ridge_grad_descent, x_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_grad_descent_train))

        # TODO: Test model on validation set
        score_mse_grad_descent_val = score_mean_squared_error(ridge_grad_descent, x_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_grad_descent_val))

        # TODO: Test model on testing set
        score_mse_grad_descent_test = score_mean_squared_error(ridge_grad_descent, x_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_grad_descent_test))
