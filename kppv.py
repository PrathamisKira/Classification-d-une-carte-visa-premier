#---------------Library----------------------------#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
#-----------------------------------------------------#


dataset = pd.read_csv('VisaPremier_f.csv') #importation
dataset.head() #affiche data

print('--------------------------------------DATA------------------------------------------------------')
print(dataset)
print('-------------------------------------------------------------------------------------------------')

print('-------------------------------------------------DATA INFO---------------------------------------')
dataset.info()

#----------- variable cible et les attribus et test-----------------#
X = dataset.drop(['cartevpr'],axis=1)
y = dataset['cartevpr']
test = ['cartevpr']
#----------------------------------------#



print("---------------------------NORMALISATION ---------------------------------------------------------")
# standardize the data attributes
standardized_X = preprocessing.normalize(X)
print("Data normalisée = \n ", standardized_X )
print('-------------------------------------------------------------------------------------------------')

# ----------------separation du Data---------------------------#
X_train, X_test, y_train, y_test = train_test_split(standardized_X, y, test_size=0.2, random_state=0)
#---------------------------------------------------------------#

#-----------Classifier implementing the k-nearest neighbors vote------------------#

knn = KNeighborsClassifier(n_neighbors=10) #Nombre de voisins à utiliser =10
digit_knn=knn.fit(X_train, y_train)
#-----------------------------------------------------------#

# -----------------grille de valeurs-------------------------#
param=[{"n_neighbors":list(range(1,15))}]
knn= GridSearchCV(KNeighborsClassifier(),param,cv=5,n_jobs=-1)
digit_knn=knn.fit(X_train, y_train)  #Ajuster le modèle
# paramètre optimal
digit_knn.best_params_["n_neighbors"]

knn = KNeighborsClassifier(n_neighbors=digit_knn.best_params_["n_neighbors"])
digit_knn=knn.fit(X_train, y_train)
#----------------------------------------------------------------#

#------------------Prévision-----------#
y_chap = digit_knn.predict(X_test)
print( "prédiction  = ", y_chap)
print('-------------------------------------------------------------------------------------------------')

#---------- Estimation de l’erreur de prévision sur l’échantillon test-----------#
erreur = 1-digit_knn.score(X_test,y_test)
print( " Erreur de prévision  = ", erreur)
print('-------------------------------------------------------------------------------------------------')

#-----mesurer les performances de modèle------------------------------#

# ------------------matrice de confusion----------------#
table=pd.crosstab(y_test,y_chap)
print("la matrice de confusion \n",table)
#-------------------------------------------------------#


# ------------------------Rapport text----------------------------------------------------------------------#

print('---------------------------------------RAPPORT-----------------------------------------------------\n')
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_chap, target_names=target_names))
print('--------------------------------------FIN -----------------------------------------------------------')

#-------------visualisation Graphique----------------------------#
plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()
#-----------------------------------------------------------------#

