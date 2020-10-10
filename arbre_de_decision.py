# ---------------------LIBRARY----------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydot

#-----------------------------------------------------------------------------------------------------------#
dataset = pd.read_csv('VisaPremier_f.csv') #importation
dataset.head() #affiche data

print('--------------------------------------DATA--------------------------------------------------------\n')
print(dataset)

print('---------------------------------------------------DATA INFO----------------------------------------')
dataset.info()
print('--------------------------------------------------------------------------------------------------\n')

#----------- variable cible et les attribues---------------------#

X = dataset.drop(['cartevpr'],axis=1)
y = dataset['cartevpr']

print("--------------------------NORMALISATION-------------------------------------------------------------")
# standardize the data attributes

standardized_X=preprocessing.normalize(X)
print("Data normalisée = \n ", standardized_X )

print("-----------------------------------------------------------------------------------------------------")


#--------------------------------------------------Decoupage en deux-----------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(standardized_X, y, test_size=0.2, random_state=0)

#-------------------------------------------------------------------------------------------------------------#

#--------------------------------IMPLIMENTAION DE L'ALGO-------------------------------------------------------#


clf = DecisionTreeClassifier(max_depth= 3)
clf_tree= clf.fit(X_train, y_train)
clf_tree.score(X_test,y_test)


#-----------------------------prévision de l’échantillon test-------------------------------------------------#
y_chap = clf_tree.predict(X_test)

#-------------------------------------------matrice de confusion-----------------------------------------------#
table=pd.crosstab(y_test,y_chap)
print(table)
#-----------------------------------Enregistrer l'arbre de decision sous format png-----------------------------#
dot_data = StringIO()
export_graphviz(clf_tree, out_file=dot_data)
(graph,)=pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree.png")
print("-----------------------------------------FIN-----------------------------------------------------------\n")

#-----------------COMMENTAIRE---------------------------#
# Pour visualiser l'arbre sur Ubuntu il faut installer le package  pydot #