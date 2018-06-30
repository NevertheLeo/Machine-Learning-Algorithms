from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
#    print('y_true',y_true)
#    print('y_pred',y_pred)
    assert len(y_true) == len(y_pred) 
#    raise NotImplementedError
    y_reduce=list(map(lambda x: x[0]-x[1],zip(y_pred,y_true)))

#    print(y_reduce)

    mse=sum(np.power(y_reduce,2))/len(y_reduce)
#    print(mse)
       
    
    return mse

def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
#    raise NotImplementedError
    true_labels=np.asanyarray(real_labels)
    pred_labels=np.asanyarray(predicted_labels)
    """TP: predict:1,true:1"""     
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    
#    """TN: predict:0,true:0""" 
#    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    
    """FP: predict:1,true:0""" 
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    
    """FN: predict:0,true:1""" 
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
 
#    print (TP,FP,TN,FN)
    
    if TP==0 or (TP+FP==0) or (TP+FN==0):
        F1=0
    else:
        Precision=TP/(TP+FP)
        Recall=TP/(TP+FN)
#       print(precision)
#       print(recall)
        F1=2*Precision*Recall/(Precision+Recall)
#       print(F1)
    return F1


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
#    raise NotImplementedError
    poly_features=[]
    for feature in features:
        poly_feature=[]
        for f in feature:
            for i in range(1,k+1):
#                print('i',i)
                poly_f=float("%.6f" % np.power(f,i))
#                print(poly_f)
                poly_feature.append(poly_f)
#                print(poly_feature)
        poly_features.append(poly_feature)
#    print (poly_features)
    return poly_features


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
#    raise NotImplementedError
    d_reduce=list(map(lambda x: x[0]-x[1],zip(point1,point2)))    
    d_sum=sum(np.power(d_reduce,2))
    return np.sqrt(d_sum)

def inner_product_distance(point1: List[float], point2: List[float]) -> float:
#    raise NotImplementedError
    d_inner=list(map(lambda x: x[0]*x[1],zip(point1,point2)))    
    return sum(d_inner)

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
#    raise NotImplementedError
    d_reduce=list(map(lambda x: x[0]-x[1],zip(point1,point2)))
    e=(-1/2)*np.power(d_reduce,2)
    return sum(-np.exp(e))

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
#        raise NotImplementedError
        new_features=[]
        for feature in features:
            new_feature=[]
            fea_scale=float("%.6f" % np.sqrt(sum(list(map(lambda x:np.power(x,2),feature)))))           
            for fea in feature:
                if fea==0:
                    new_feature.append(0)
                else:
                    new_feature.append(fea/fea_scale)
#            print(new_feature)
            new_features.append(new_feature)
#       print(new_features)
        return new_features
                
                


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    count=0
    max_col=0
    min_col=0
    
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
#        raise NotImplementedError
#        count=1
        
        MinMaxScaler.count+=1
#        self.count=self.count+1
        if MinMaxScaler.count==1:
            
            fea_array=np.array(features)
        
            MinMaxScaler.max_col=fea_array.max(0)
#        print(max_col)
            MinMaxScaler.min_col=fea_array.min(0)
#        print(min_col)                
            new_features=[]
            for feature in features:
                new_feature=[]
                for i in range(len(feature)):
                    f=(feature[i]-MinMaxScaler.min_col[i])/(MinMaxScaler.max_col[i]-MinMaxScaler.min_col[i])
                    new_feature.append(float("%.6f" % f))
                new_features.append(new_feature)
#            print(new_features)
            
            
        else:
                
            new_features=[]
            for feature in features:
                new_feature=[]
                for i in range(len(feature)):
                    f=(feature[i]-MinMaxScaler.min_col[i])/(MinMaxScaler.max_col[i]-MinMaxScaler.min_col[i])
                    new_feature.append(float("%.6f" % f))
                new_features.append(new_feature)
#            print(new_features)
        return new_features
