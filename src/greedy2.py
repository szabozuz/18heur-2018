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

    def generate_problem(self, a , minweight, maxweight, minprofit, maxprofit, minTW, maxTW):
        """
        Generates array for knapsack problem
        :param a: number of items/points
        :param minweight: minimum weight of items
        :param maxweight: maximum weight of items
        :param minprofit: minimum profit 
        :param maxprofit: maximum profit 
        :return: list
        """
        
        return random.sample(range(minweight, maxweight), a), random.sample(range(minprofit, maxprofit), a), np.random.randint(minTW, maxTW, size = 1)[0]
    
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
    
    def KnapsackSA(self, value, weight, TotalWeight, beta, T0, cmax):
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
        c = 0
        st = 0
        T = T0
        temperature = []
        v = len(value)
        X = np.zeros(v)
        Y = np.zeros(v)
        CurW = 0
        bestX = np.array(X)
        while c <= cmax:
            j = np.random.randint(0, v, size=1)[0]
            Y = np.array(X)
            Y[j] = 1 - X[j]
            if Y[j] == 0 or CurW + weight[j] <= TotalWeight:
                if Y[j] == 1:
                    X = np.array(Y)
                    CurW = CurW + weight[j]
                    if sum(X * value) > sum(bestX * value):
                        bestX = np.array(X)
                else:
                    r = random.uniform(0, 1)
                    if r < math.exp((sum(Y * value) - sum(X * value)) / T):
                        X = np.array(Y)
                        CurW = CurW - weight[j]
            c = c + 1
            T = beta * T
            temperature.append(T)
            if sum(bestX * value) != 200:
                st += 1
            
        return bestX, sum(bestX * value), temperature, st
    
    def KnapsackTS(self, value, weight, TotalWeight, cmax, L):
        """
        Solves the 0/1 Knapsack problem by means of tabu search
            Input: value - list of values for each item
                   weight - list of weights for each item
                   TotalWeight - weight limit
                   cmax - number of steps
                   L - number of repetitions
            Output: FinalItems - ids of the items added to the Knapsack
                    FinalValue - value of the items in the Knapsack
        """
        value, weight = self.shuffleWeightsValues(value, weight)

        c = 1

        v = len(value)
        X = np.zeros(v)

        X = self.randomFeasibleSolution(X, weight, TotalWeight)

        CurW = sum(X*weight)
        bestX = np.array(X)
        tabuList = []
        
        tabu = np.random.randint(0, v, size=1)[0]
        tabuList.append(tabu)
        
        while c <= cmax:
            N = list(range(v))
            start = max([0, c-L])
            if tabuList != []:
                for j in range(start, c):
                    N.remove(tabuList[j])
            rm = []
            for i in N:
                if X[i] == 0 and CurW + weight[i] > TotalWeight:
                    rm.append(i)
            N = [x for x in N if x not in rm]
            if N == []:
                break
            r = []
            for k in N:
                r.append(np.power(-1, X[k]) * value[k] / weight[k])
                
            i = N[np.argmax(np.array(r))]
            tabuList.append(i)
            X[i] = 1- X[i]
            if X[i] == 1:
                CurW = CurW + weight[i]
            else:
                CurW = CurW - weight[i]
            if sum(X*value)>sum(bestX*value):
                bestX = np.array(X)
            c += 1
        return bestX, sum(bestX*value), value, weight
    
    def shuffleWeightsValues(self, a, b):
        """ shuffles weight and value list with the same order
        
            INPUT: a, b as lists
            OUTPUT: a, b as shuffled lists
        """
        
        c = list(zip(a,b))
        random.shuffle(c)
        a,b = zip(*c)

        return(a,b)
    
    def randomFeasibleSolution(self, X, weight, TotalWeight):
        """ finds a random feasible solution to knapsack problem using shuffled
            list of weights and values/profits
            
            INPUT: X - binary array including 1 if item is added to knapsack and 0 if 
                   item is exluded
                   weight - list of weights
                   TotalWeight - weight limit for knapsack
            OUTPUT: X - some feasible solution for Knapsack problem
        """
        
        suma = 0
        for k in range(len(weight)):
            if suma + weight[k] <= TotalWeight:
                X[k] = 1
                suma += weight[k]

        return X