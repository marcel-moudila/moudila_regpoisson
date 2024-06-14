import numpy as np
import pandas as pd
from math import exp,sqrt, log
from scipy.stats import chi2,norm
from numpy.linalg import inv, multi_dot


from sklearn.linear_model import PoissonRegressor
def param(y,x):
    reg = PoissonRegressor(alpha=0)
    reg.fit(x, y)
    beta =[reg.intercept_]+[reg.coef_[i] for i in range(len(reg.coef_))]
    return beta

def matrice_X(x):
    # x : vecteur des covariables 
    # X : matrice des donnees des covariables 
    n = len(x)
    p = len(x[0])
    X = np.zeros((n,p+1))
    col = np.ones((n,1))
    for i in range(n):
        X[i] = np.concatenate((col[i],x[i]))
    return X

def lambda_x(beta, X):
    # beta : beta(beta_0,...,beta_p) 
    # X : matrice des donnees des covariables 
    # sortie : lambda_x(lambda (x_1),...lambda(x_n))
    n = len(X)
    lambda_x = np.zeros(n)
    for i in range(n):
        lambda_x[i] = np.exp(beta@X[i])
    return lambda_x

def var_b(ld,X):
    # ld : ld(lambda (x_1),...lambda(x_n))
    # X : matrice des données des covariables
    # sortie : matrice des covariances/variances de beta
    n = len(X)
    W = np.diag(ld)
    var = inv(multi_dot([np.transpose(X),W,X]))
    return var

def ecarttype(V):
    # V:  variance de beta
    # sortie : ecart type sd(sd(beta_0),..sd(beta_p))
    
    n = len(V)
    sd = np.zeros(n)
    for i in range(n):
        sd[i] = sqrt(V[i,i])
    return sd

def ICbeta(beta, alpha, sd):
    # beta : coefficients du modele
    # alpha : seuil de confiance
    # sd : vecteur des ecart-type de beta
    # sortie : tableau des intervalles de confiance de beta_j
    
    z_alpha = norm.ppf(1-(alpha/2),0,1)
    p = len(beta)-1
    intervalle = np.zeros((p+1,2))
    for i in range(p+1):
        intervalle[i]= np.array([beta[i]-z_alpha*sd[i],
                                beta[i]+z_alpha*sd[i]])
    resultat = pd.DataFrame(intervalle, 
               columns=['borne inf','borne sup'],
               index = ["beta_"+str(i) for i in range(p+1)])
    return resultat

def testWald(beta,sd,n):
    # beta : coefficients du modele
    # sd : vecteur des ecart-type de beta
    # n : nombre d'individus
    # sortie : un tableau des resultats du test de Wald
    p = len(beta)-1
    tab = np.zeros((p+1,4))
    Zobs = np.zeros(p+1)
    pvalue = np.zeros(p+1)
    for i in range(p+1):
        Zobs[i]= beta[i]/sd[i]
        pvalue[i]= (1-norm.cdf(abs(Zobs[i]),0,1))*2
        tab[i] = np.array([beta[i],sd[i],Zobs[i],pvalue[i]])
    resultat = pd.DataFrame(tab, 
               columns=['coefficients','ecart-type','statistique de test','p-value'],
               index = ["beta_"+str(i) for i in range(p+1)])
    return resultat

def residu_pearson(y,ld,beta):
    # y : donnees de la avariable cible
    # ld : ld(lambda (x_1),...lambda(x_n))
    # beta : coefficients du modele
    # sortie : les residus de pearson
    
    n = len(y)
    p = len(beta)
    rsp = np.zeros(n)
    lambda_x = ld
    for i in range(n) :
        rsp[i] = np.array((y[i] - lambda_x[i])/sqrt(lambda_x[i]))    
    return rsp

def residu_deviance(y,ld,beta):
    # y : donnees de la avariable cible
    # ld : ld(lambda (x_1),...lambda(x_n))
    # beta : coefficients du modele
    # sortie : les residus de deviance
    
    n = len(y)
    rsd = np.zeros(n)
    lambda_x = ld
    for i in range(n) :
        if y[i]== 0:
           rsd[i] = np.sign(y[i]-lambda_x[i])*sqrt(2*(y[i]-(y[i]-lambda_x[i])))
        if y[i] != 0:   
           rsd[i] = np.sign(y[i]-lambda_x[i])*sqrt(2*(y[i]*log(y[i]/
                 lambda_x[i])-(y[i]-lambda_x[i])))
    return rsd

def pertinence(y,resd,resp,beta):
    # y : donnees de la avariable cible
    # resd : les residus de deviance
    # resp : les residus de pearson
    # beta : coefficients du modele
    # sortie : tableau de pertinence 
    n = len(y)
    p = len(beta)
    #deviance
    Zobs_d = np.sum(resd**2)
    ddl = n-(p+1)
    pvalue_d = 1 - chi2.cdf(Zobs_d,ddl)
    #pearson
    Zobs_p = np.sum(resp**2)
    ddl = n-(p+1)
    pvalue_p = 1 - chi2.cdf(Zobs_p,ddl)
    tab = pd.DataFrame(np.array([[Zobs_d,pvalue_d ],[Zobs_p,pvalue_p]]), 
               columns=['statistique de test','p-value'],
               index = ["deviance","pearson"])
    return tab

def residu_pearson_standardise(resp,ld,X,V):
    # resp : les residus de pearson
    # ld : ld(lambda (x_1),...lambda(x_n))
    # X : matrice des données des covariables
    # V : matrice des covariances/variances de beta
    
    # sortie : un tableau des observations aberrantes
    n = len(resp)
    W = np.diag(ld)**(1/2)
    mat = multi_dot([W,X,V,np.transpose(X),W])
    resp_stand = np.zeros(n)
    for i in range(n):
        resp_stand[i] = resp[i]/sqrt(1-mat[i][i])
    return resp_stand

def aberrantes(resp_stand):
    # resp_stand : les résidus standardises de pearson
    # sortie : la liste des observations aberrantes
    
    n = len(resp_stand)
    liste = []
    for i in range(n):
        if abs(resp_stand[i]) > 2:
            liste = liste +[i]
    return liste

def moudilaRegpoisson(y,x,alpha):
    
    beta = param(y,x)                              # coeff du modele
    X = matrice_X(x)                               # matrice X
    ld = lambda_x(beta,X)                          # lambda
    V = var_b(ld,X)                                # variance de beta
    sd = ecarttype(V)                              # ecart-type de beta_j
    IC = ICbeta(beta,alpha,sd)                     # intervalle de confiance de beta_j
    n = len(ld)
    tab = testWald(beta,sd,n)                      # test de nullité de beta_j
    resp = residu_pearson(y,ld,beta)               # residu de pearson
    resd = residu_deviance(y,ld,beta)              # residu de deviance
    tab2 = pertinence(y,resd,resp,beta)            # test sur la pertinence du modèle
    res = residu_pearson_standardise(resp,ld,X,V)  # résidus standardisés de pearson
    tab3 = aberrantes(res)                         # valeurs anormales
    
    dictionnaire = {"coeff":beta,"matrice": X,"lambda" :ld,"variance" :V,"ecarttype":sd,
                    "intervalle.confiance":IC,"test.wald":tab,"residu.pearson":resp,
                    "residu.deviance":resd,"pertinence":tab2,
                    "residu.pearson.standard":res,"anormale":tab3}
                    
    return dictionnaire

def pred_Moud_reg_pois(dictionnaire,x):
    beta = np.array(dictionnaire["coeff"])
    xnew = np.array([1]+[i for i in x])
    if len(beta) != len(x)+1 :
        pass
    else :   
        return np.exp(beta@xnew)

def affichage(dictionnaire) :
    
    for cle,valeur in dictionnaire.items():
        print (cle)
        print()
        print(valeur)
        print()
        print("=======================================================")
    return "implemente par Marcel MOUDILA---M1 ingenierie mathematiques----Nancy"



