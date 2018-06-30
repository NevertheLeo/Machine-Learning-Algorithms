from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy



############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.features=features
        self.labels=labels
        
#        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        
        pred_result=[]   
#        print()
        for new_fea in features:
#            print(new_fea)
            distance=[]
            for x in range(len(self.features)):
#                print('train:',self.features[x])
                dis=self.distance_function(new_fea,self.features[x])
#                print('dis:',dis)
                distance.append((x,dis))
#            print(distance)
                """use 'operator' to sort by distance"""
            import operator
            distance.sort(key=operator.itemgetter(1))
#            print('dis_sort',distance)
            neighbors = []
            for i in range(self.k):
                neighbors.append(distance[i][0])
#                print(neighbors)
            knn_labels=[]
            for j in neighbors:
                knn_labels.append(self.labels[j])
#            print(knn_labels)
            pred_result.append(max(knn_labels, key=knn_labels.count))
#            print(pred_result) 
        return pred_result
                





if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)