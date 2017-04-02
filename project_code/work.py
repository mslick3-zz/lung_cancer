import numpy as np

shape = (1000,1000)
a = np.random.randint(0,2,shape)
v,c=np.unique(a, return_counts=True)
print(v)
print(c/np.sum(c))
