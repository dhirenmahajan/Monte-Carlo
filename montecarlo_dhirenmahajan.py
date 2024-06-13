# -*- coding: utf-8 -*-
"""
Created on Thur Dec  1 21:09:54 2022

@author: dhirenmahajan
"""

import numpy as np

from random import random

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
 

# Generate Data

balls = np.arange(1,1001)

count_nonemptybin=np.arange(1,1001)

for N in balls: 
  bins = np.zeros(N) 
  for b in range(N): 
    bins[int(N * random())] +=1 
  count=0
  for i in bins:
      if(i!=0):
          count+=1
  count_nonemptybin[N-1]=count


#Plotting
plt.plot(balls, count_nonemptybin,'g') 
plt.xlabel("Number of Balls")
plt.ylabel("Number of Non Empty Bins")
plt.show() 

#LinearRegression usage
Model = LinearRegression()
Model.fit(balls.reshape(-1, 1),count_nonemptybin.reshape(-1, 1))
intercept = Model.intercept_
print("slope: ", Model.coef_[0][0])
print("intercept: ",intercept[0])
print("r value: ",Model.score(balls.reshape(-1,1), count_nonemptybin.reshape(-1,1)))

"""output
slope:  0.6328644328644325
intercept:  -0.27264864864844185
r value:  0.998624227905292
"""


