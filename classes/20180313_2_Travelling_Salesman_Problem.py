
# coding: utf-8

# # Travelling Salesman Problem (TSP) on a rectangular grid 
# 
# ### Objective function description
# 
# * Cities placed on rectangular grid in $\mathbb{R}^n$, dimension given by $A, B \in \mathbb{N}$
# * Assuming Euclidean distance the optimal tour has length  
#   * $A \cdot B + \sqrt{2} - 1$ if $A$ and $B$ are even numbers
#   * $A \cdot B$ otherwise
# 
# **How to find optimal tour using heuristics?**
# * Our success depends on efficient solution encoding
#   * Rather extreme, binary, representation would result in $2^{n^2}$ states
#   * Let's investigate an encoding using $(n-1)!$ states "only", as demonstrated on the following example with $3 \times 2$ grid:
#   
# <img src="img/tsp_example.png">
# 
# **Notes**:
# 
# * $n = A \cdot B$
# * $\mathbf{a} = (0, 0, \ldots, 0)$ 
# * $\mathbf{b} = (n-2, n-3, \ldots, 0)$
# * $f^*$ quals to 
#   * $A \cdot B + \sqrt{2} - 1$ if both $A$ and $B$ are even numbers
#   * $A \cdot B$ otherwise
# * *For serious TSP optimization you should use much more sophisticated approaches, e.g. the [Concorde](https://en.wikipedia.org/wiki/Concorde_TSP_Solver) algorithm*

# # Implementation
# 
# You can find it in `src/objfun_tsp.py`, class `TSPGrid`.
# 
# Real-world demonstration follows:

# In[1]:

# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().magic('pwd')
sys.path.append(os.path.join(pwd, os.path.join('..', 'src')))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[2]:

# Import extrenal librarires
import numpy as np

# Import our code
from heur_sg import ShootAndGo
from objfun_tsp import TSPGrid  # <-- our implementation of TSP


# ### ``TSPGrid(3, 2)`` demonstration

# In[3]:

# initialization
tsp = TSPGrid(3, 2)


# In[4]:

# random point generation
x = tsp.generate_point()
print(x)


# In[5]:

# decode this solution (into list of visited cities)
cx = tsp.decode(x)
print(cx)


# In[6]:

# what is the cost of such tour?
of_val = tsp.evaluate(x)
print(of_val)


# In[7]:

# neighbourhood of x:
N = tsp.get_neighborhood(x, 1)
print(N)


# In[8]:

# decoded neighbours and their objective function values
for xn in N:
    print('{} ({}) -> {:.4f}'.format(xn, tsp.decode(xn), tsp.evaluate(xn)))


# **Carefully** mind the difference between encoded solution vector vs decoded city tour and meaning of such neighbourhood.

# ### TSP optimization using Random Shooting ($\mathrm{SG}_{0}$)

# In[9]:

heur = ShootAndGo(tsp, maxeval=1000, hmax=0)
print(heur.search())


# # Assignments:
# 
# 1. Find a better performing heuristic (to test TSP implementation on your own).
# 2. Can you improve heuristic performance using any 
#    1. **better random point generator**?
#    2. **better neighbourhood generator**?
# 
# Use performance measure(s) of your choice (e.g. $FEO$).
