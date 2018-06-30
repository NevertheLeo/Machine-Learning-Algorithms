from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=0):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        self.nb_features=nb_features
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
#        raise NotImplementedError
        
        self.features=features
#        print(self.features)
        self.labels=labels
#        print(self.labels)
               
#        y=numpy.mat(self.values)
        
        epsilon=1e-3
        while(self.max_iteration):
            count=0
            for i in range(len(self.features)):               
                """calculate ||x|| and ||w||"""               
                x_i_mode=np.sqrt(sum(list(map(lambda x:np.power(x,2),self.features[i]))))
#            print(x_i_mode)
            
                w_mode=np.sqrt(sum(list(map(lambda xx:np.power(xx,2),self.w))))
#                print(w_mode)     
                """convert lists into matrix(e.g. self.features and self.w)"""
#            print(self.features[i])
                x_train=np.mat(self.features[i])
                weight=np.mat(self.w)
                """update weights"""
                
                if np.sign(weight*x_train.T)*self.labels[i]!=1:
                    
    #                print('y')
                    count+=1
                    ww=[]
                    for x in self.features[i]:
    #                    print(x)
                        ww.append(self.labels[i]*x/x_i_mode)
    #                self.w+=self.labels[i]*x_train/x_i_mode
                    self.w=list(map(lambda x: x[0]+x[1],zip(self.w,ww)))
#                    print(self.w)
    
                elif self.margin!=0:
#                    print('?',weight*x_train.T/(w_mode+epsilon))
                    if weight*x_train.T/(w_mode+epsilon) >(-self.margin/2) and weight*x_train.T/(w_mode+epsilon)<(self.margin/2):
                        count+=1
                        ww=[]
                        for x in self.features[i]:
    #                        print(x)
                            ww.append(self.labels[i]*x/x_i_mode)
                        self.w=list(map(lambda x: x[0]+x[1],zip(self.w,ww)))                    
#                        print(self.w)                 
                else:
                    continue
#            print('each loop',count)
            if count==0:
                return 'True'
            else:
                self.max_iteration-=1
#        print(count)
        return 'False'
             
            
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        
#        raise NotImplementedError
        """convert list into matrix"""
        x_test=np.mat(features)
#        print(X_.shape)
        w_=self.get_weights()
        w=np.mat(w_)
        
        
        y_pred=np.sign(w*x_test.T)
#        print(y_pred)
        y_pred_list=y_pred.tolist()
#        print(y_pred_list[0])
#        raise NotImplementedError
        return y_pred_list[0]

    def get_weights(self) -> List[float]:
        return self.w
    
