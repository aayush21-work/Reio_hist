import numpy as np

fil=np.loadtxt("chains/tau_mcmc.1.txt")

best=np.argmin(fil[:,1])

print(fil[best,2])
print(fil[best,3])
print(fil[best,4])





