# ---------------------LIBRARY----------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
#--------------------------------------------------------------#

dataset = pd.read_csv('VisaPremier_f.csv') #importation
dataset.head() #affiche data
#---------------------------------------------------------------#


print('--------------------------------------DATA--------------------------------------------------------\n')
print(dataset)

print('---------------------------------------------------DATA INFO----------------------------------------')
dataset.info()
print('--------------------------------------------------------------------------------------------------\n')

#----------- variable cible et les attribues---------------------#
X = dataset.drop(['cartevpr'],axis=1)
y = dataset['cartevpr']
#----------------------------------------------------------------#


print("--------------------------NORMALISATION-------------------------------------------------------------")
# standardize the data attributes
standardized_X = preprocessing.normalize(X)
print("Data normalisée = \n ", standardized_X )

print("-----------------------------------------------------------------------------------------------------")


#--------------------------------------------------Decoupage en deux-----------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(standardized_X, y, test_size=0.2, random_state=0)
#-------------------------------------------------------------------------------------------------------------#

#--------------------------------IMPLIMENTAION DE L'ALGO-------------------------------------------------------#
n_estimators=[]
for i in range(1,40):
    clf = RandomForestClassifier(n_estimators=i, random_state=0)
    clf.fit(X_train, y_train)
    i_pred = clf.predict(X_test)
    n_estimators.append(np.mean(i_pred != y_test))
#----------------------------------------------------------------------------------------------------------------#

#-------------------------visualisation du graphe-----------------------------------------------------------------#
plt.figure(figsize=(40,1))
plt.plot(range(1,40),n_estimators,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize='10')
plt.xlabel('number of n_estimators')
plt.ylabel('Error Rate')
plt.show()
#-----------------------------------------------------------------------------------------------------------------#



#----------------------------------EVALUATION----------------------------------------------------------------------#
temps1=time.time()
rf = RandomForestClassifier(n_jobs = -1, n_estimators=7,max_features='auto',min_samples_leaf=1)
print("le temps d'apprentissage =", temps1)
print("-----------------------------------------------------------------------------------------------------\n")


prediction = rf.fit(X_train,y_train)
tmps1= time.time()
print("effectuer la prediction \n", rf.predict(X_test))
print("-----------------------------------------------------------------------------------------------------\n")

y_pred = rf.predict(X_test)
tmps2 = time.time()-tmps1
print("le temps de prediction =",tmps2)
print("-----------------------------------------------------------------------------------------------------\n")

error_predict = np.sqrt(((y_pred-y_test)**2).mean())
print("le taux d'erreur ", error_predict)
print("-----------------------------------------------------------------------------------------------------\n")
pscore = accuracy_score(y_test,y_pred)*100
print("le score est de",pscore)
print("-----------------------------------------------------------------------------------------------------\n")

#Créer un rapport texte
print("----------------------------------------RAPPORT-------------------------------------------------------\n")
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
print("-----------------------------------------FIN-----------------------------------------------------------\n")
