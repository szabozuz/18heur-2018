from objfun import ObjFun
import numpy as np
import random
from operator import itemgetter, attrgetter
import math


class GreedyAlg2(ObjFun):
    """
    Heuristics: greedy algorithm
    """

    def __init__(self):
        """

        """

    def generate_problem(self, a , minweight, maxweight, minprofit, maxprofit):
        """
        Generates array for knapsack problem
        :param a: number of items/points
        :param minweight: minimum weight of items
        :param maxweight: maximum weight of items
        :param minprofit: minimum profit 
        :param maxprofit: maximum profit 
        :return: list
        """
        
        return [random.sample(range(minweight, maxweight), a),random.sample(range(minprofit, maxprofit), a)]
    
    def knapsack_fractional(self, W, weights, values):
        
        """
        Returns the maximum value that can be put in a knapsack of capacity W
            Input: W - weight limit
                   weights - list of weights for each item
                   values - list of values for each item
            Output: value - value of the items in the Knapsack
        """
        
        capacity = 0
        value = 0.

        fractions = [[v / w, w] for v,w in zip(values,weights)]
        valuePerWeight = sorted(fractions, reverse=True)
        
        while W > 0 and valuePerWeight:
            maxi = 0
            idx = None
            for i,item in enumerate(valuePerWeight):
                if item [1] > 0 and maxi < item [0]:
                    maxi = item [0]
                    idx = i

            if idx is None:
                return 0.

            v = valuePerWeight[idx][0]
            w = valuePerWeight[idx][1]

            if w <= W:
                value += v*w
                W -= w
            else:
                if w > 0:
                    value += W * v
                    return value
            valuePerWeight.pop(idx)

        return value

    
    def greedy_0_1_dynamic(self, W, weights, values):
        
        """
        Returns the maximum value that can be put in a knapsack of capacity W
            Input: W - weight limit
                   weights - list of weights for each item
                   values - list of values for each item
            Output: FinalValue - value of the items in the Knapsack
        """
        
        A = np.zeros((len(weights)+1, W +1));

        for j in range(len(weights)+1):
            for i in range(W+1):
                if j ==0 or i == 0:
                    A[j, i] = 0
                elif weights[j-1] > i:
                    A[j, i] = A[j-1, i];
                else:
                    A[j, i] = max([A[j-1, i], values[j-1] + A[j-1, i - weights[j-1]]]);

        FinalValue = A[-1,-1];
        return FinalValue,A
    
    def BackTracking(self, W, weights, A):
        
        """
        Returns the list of items to be put into the knapsack of capacity W, solved by function greedy_0_1_dynamic
            Input: W - weight limit
                   weights - list of weights for each item
                   A - matrix of values from greedy_0_1_dynamic function
            Output: items - sorted items to be added to the knapsack
        """
    
        i = len(weights);
        items = [];

        while i > 0 and W > 0:
            if A[i, W] != A[i - 1, W]:
                items.append(i-1);
                W += -weights[i-1];
                i += -1;
            else:
                i += -1;

        return sorted(items)
    
    
    def Knapsack(self,value, weight, TotalWeight, beta, n):
        
        """
        Solves the 0/1 Knapsack problem by means of simulated annealing
            Input: value - list of values for each item
                   weight - list of weights for each item
                   TotalWeight - weight limit
                   beta - parameter describing the cooling
                   n - number of iterations
            Output: FinalItems - ids of the items added to the Knapsack
                    FinalValue - value of the items in the Knapsack
        """
        
        v = len(value)
        W = np.zeros(v)
        Value = [0]
        VW = [0]*1000

        for i in range(len(beta)):
            b = beta[i]
            for j in range(1, int(n) ):
                c = self.getRandomItemFromItemSet(W,v)

                Wnew = np.array(W)
                Wnew[c] = 1

                while self.exceeds(sum(Wnew * weight),TotalWeight):
                    d = self.RemoveRandomItemFromItemSet(Wnew, v)
                        
                    Wnew[d] = 0;

                delta = sum(Wnew * value) - sum(W * value)
                g = min([1, math.exp(b * delta)])

                if np.random.rand() <= g:
                    W = np.array(Wnew)
                    VW[j] = sum(W * value)
                else:
                    VW[j] = VW[j - 1]

            Value = Value + VW[1:]

        FinalValue = Value[-1]
        x = [0]

        FinalItems = [i for i,x in enumerate(list(W)) if x==1]

        return FinalItems, FinalValue
    
    def exceeds(self, x, limit):
        """
        Checks whether the current selection is overweight or not
        """
        return x>limit
    
    def getRandomItemFromItemSet(self, ItemSet, NumItems):
        """
        Returns any random item from the itemset which is not in the currentSolution of the bag
        """
        c = np.random.randint(NumItems)
        while ItemSet[c] == 1:
            c = np.random.randint(NumItems)
        return c
    
    def RemoveRandomItemFromItemSet(self, ItemSet, NumItems):
        """
        Returns any random item to be removed from the itemset which is in the currentSolution of the bag
        """
        c = np.random.randint(NumItems)
        while ItemSet[c] == 0:
            c = np.random.randint(NumItems)
        return c
        