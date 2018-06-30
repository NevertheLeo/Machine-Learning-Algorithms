from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""

        self.features=features
        self.values=values
        
#        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        X_test=numpy.mat(features)
        one=numpy.ones((len(X_test),1))
        X_=numpy.column_stack((one,X_test))
#        print(X_.shape)
        w_=self.get_weights()
        w=numpy.mat(w_)
        y_pred=w*X_.T
#        print(y_pred)
        y_pred_list=y_pred.tolist()
#        print(y_pred_list[0])
#        raise NotImplementedError
        return y_pred_list[0]


    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        
        weights=numpy.zeros(len(self.features))
        
        X_train=numpy.mat(self.features)        
        y=numpy.mat(self.values)
        """w=(X_'*X)^(-1)*X'*y'"""
        
        """x_=[1 X]"""
        one=numpy.ones((len(X_train),1))
        X_=numpy.column_stack((one,X_train))
        """X.transpose"""
        X_T=X_.T
#        print(X_T.shape)
        """inverse"""
        X_inv=(X_T*X_).I
#        print(X_inv.shape)
        weights=X_inv*X_T*y.T
        w=weights.T
        w_=w.tolist()[0]

#        print(w_)
#        raise NotImplementedError
        return w_


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
#        raise NotImplementedError
        self.features=features
        self.values=values

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
#        raise NotImplementedError
        X_test=numpy.mat(features)
        one=numpy.ones((len(X_test),1))
        X_=numpy.column_stack((one,X_test))
#        print(X_.shape)
        w_=self.get_weights()
        w=numpy.mat(w_)
        y_pred=w*X_.T
#        print(y_pred)
        y_pred_list=y_pred.tolist()
#        print(y_pred_list[0])
#        raise NotImplementedError
        return y_pred_list[0]

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
#        raise NotImplementedError
        weights=numpy.zeros(len(self.features))
        
        X_train=numpy.mat(self.features)        
        y=numpy.mat(self.values)
        """w=(X_'*X)^(-1)*X'*y'"""
        
        """x_=[1 X]"""
        one=numpy.ones((len(X_train),1))
        X_=numpy.column_stack((one,X_train))
        n=X_.shape[1]
        """X.transpose"""
        X_T=X_.T
#        print(X_T.shape)

        reg=X_T*X_+ self.alpha*numpy.eye(n)
        """inverse"""
        X_inv=reg.I
#        print(X_inv.shape)
        weights=X_inv*X_T*y.T
        w=weights.T
        w_=w.tolist()[0]

#        print(w_)
#        raise NotImplementedError
        return w_


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
