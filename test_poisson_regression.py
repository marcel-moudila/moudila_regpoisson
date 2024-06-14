import numpy as np
import moudilaCode

y = np.array([2,4,1,2,1])
x = np.array([[7,12],[3,2],[-2,5],[1,4],[2,3]])
alpha = 0.05
dico = moudilaCode.moudilaRegpoisson(y,x,alpha)
print(moudilaCode.affichage(dico))
