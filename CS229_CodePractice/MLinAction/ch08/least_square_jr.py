import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, x, y):
        """
        :param x:<np.array>: Of shape [m,]
        :param y:<np.array>: Of shape [m,]
        """

        ones = np.ones(len(x))
        """ 
        Your code here: add a column with all ones
        Start
        """
        self.x = np.vstack((ones, x)).T
        """End"""
        self.y = y

    def loss(self, w):
        """
        :param w: <np.array>: Of shape [2,]
        """

        """ 
        Your code here
        Implement the loss function l = 1/m * sum (y_i - yhat_i) ^ 2
        Start
        """
        hypothesis = np.dot(self.x, w.T)
        loss = ((hypothesis - self.y) ** 2).sum() / (2 * self.x.shape[0])
        """End"""
        return loss

    def gradient(self, w):
        """
        Your code here
        Calculate the gradient of w : dl/dw
        Hint: You should first do the forward pass
        Start
        """
        x = self.x
        y = self.y
        hypothesis = np.dot(x, w.T)
        grad = (y - hypothesis).T.dot(x) / x.shape[0]
        """End"""
        return grad

    def fit_gradient_descent(self):
        w = np.random.randn(2) * 0.1
        eps = 1e-6
        learning_rate = 1e-3

        """
        Implement gradient descent with stopping criterion  ||w - w'|| < eps
        """
        converge = False
        num_iter = 0
        lossList = []
        while not converge:
            num_iter += 1
            w_prev = w.copy()

            """
            Your code here
            Calculate the loss under current w, print the loss
            Calculate the gradient dl/dw and apply gradient descent
            You may use self.loss and self.gradient
            Start
            """
            w += learning_rate * self.gradient(w)
            loss = self.loss(w)
            print(num_iter, loss)
            lossList.append(loss)
            """End"""

            if np.linalg.norm(w - w_prev) <= eps:
                converge = True

        plt.figure(2)
        plt.plot(np.arange(num_iter), lossList, color="c", marker='x', linestyle="-.", label='Loss Change')
        plt.legend()
        return w

    def fit_normal_equation(self):
        w = None
        """
        Your code here
        Implement normal equation w = (X^T X)^{-1}X^Ty and return w
        Start
        """
        x = self.x
        w = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), y)
        """End"""
        return w

    def predict(self, x, w):
        """
        :param x: Of shape [m,]
        :param w: Of shape [2,]
        :return:
        """

        ones = np.ones(len(x))
        """ 
        Your code here: You should add a column with all ones to X
        Start
        """
        X = np.vstack((ones, x))
        ypred = np.dot(w, X)
        """End"""
        return ypred


if __name__ == "__main__":
    seed = 42
    m = 100

    X = np.arange(0, 10, float(10) / m)

    w_true = np.random.randn(1)[0] * 5
    b_true = np.random.randn(1)[0] * 5

    # Training data is generated by y = w * x + b
    y = X * w_true + b_true + np.random.randn(m)

    # plot your training data
    plt.figure(1)
    plt.plot(X, y, 'ro', label='samples')
    plt.legend()

    # Complete the LinearRegresson class
    model = LinearRegression(X, y)
    w1 = model.fit_normal_equation()
    w2 = model.fit_gradient_descent()

    print("True weight: ", np.array([b_true, w_true]))
    print("Normal Equation fit: ", w1)
    print("Gradient Descent fit: ", w2)

    xmin = np.min(X)
    xmax = np.max(X)
    xrange = np.array([xmin, xmax])

    ypred1 = model.predict(xrange, w1)
    ypred2 = model.predict(xrange, w2)

    # Plot your result
    plt.figure(1)
    plt.plot(xrange, ypred1, 'b--', label='Normal Equation fit')
    plt.plot(xrange, ypred2, 'g-.', label='Gradient Descent fit')
    plt.legend()
    plt.show()
