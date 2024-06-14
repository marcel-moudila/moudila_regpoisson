import unittest
import numpy as np
from moudilaCode import moudilaRegpoisson, affichage

import pandas as pd
# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(42)

# Nombre de lignes dans le DataFrame
n = 1000

# Génération des données
# Paramètre de la distribution de Poisson pour Y (choisissez un nombre lambda)
lambda_param = 2.5
Y = np.random.poisson(lambda_param, n)

# Génération des variables X1 et X2
X1 = np.random.uniform(0, 10, n)  # Valeurs aléatoires uniformément distribuées entre 0 et 10
X2 = np.random.normal(0, 1, n)     # Valeurs aléatoires suivant une distribution normale

# Création du DataFrame
data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2})

# Affichage des premières lignes du DataFrame pour vérification
print(data.head())

# Séparation des variables explicatives (X) et de la variable cible (Y)
x = data[['X1', 'X2']].values
y = data['Y'].values
print(y)
print(x)



class TestPoissonRegression(unittest.TestCase):
    def test_poisson_regression(self):
        #y = np.array([2,4,1,2,1])
        #x = np.array([[7,12],[3,2],[-2,5],[1,4],[2,3]])
        alpha = 0.05
        result = moudilaRegpoisson(y, x, alpha)
        self.assertIsNotNone(result)
        print(affichage(result))

if __name__ == '__main__':
    unittest.main()


