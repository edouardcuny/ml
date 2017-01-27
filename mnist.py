# importation des modules dont on va se servir
import pandas as pd
import os
import operator

# on se place dans le bon dossier
os.chdir("/Users/edouardcuny/Desktop/10k/workshp/")

# importation des données
entrainement=pd.read_csv("train.csv")

#%%

# on crée une copie de train sinon les modifications sur x modifient aussi train
# différent des entiers

# afin de pouvoir tester sur éléments labelisés je partage mon set de données
# je vais entrainer sur 30 000 lignes et tester sur le reste
train=entrainement.copy()
train=train.ix[:30000,]

# j'ai une colonne label (y) qui donne le chiffre associé à mon observation
# je crée à partir de ça 10 nouvelles colonnes correspondant à l'output
# souhaité de mon réseau de neurones - ie que j'aurais dans la ième ligne de mon
# output la probabilité que le chiffre soit un i
x=train.copy()
for i in range(10):
    x[str(i)]=(x.label==i).astype(int)
del x['label'] # je supprime ensuite la colonne label qui ne me sert plus

# afin de pouvoir effectuer les calculs il faut que je fasse du feature scaling
x=x.as_matrix()
x=x.astype(float)
x[:,:-10]=x[:,:-10]*(1/255) # grayscale prend ses valeurs entre 0 et 255

#%% j'entraine mon réseau de neurones
# je reprends la strucutre de réseau de Michael Nielsen dans son cours
import neuralNetwork
neural_net=neuralNetwork.NeuralNetwork([784,30,10])
neural_net.SGD(x,batch_size=10,epochs=5)

#%% TEST - évaluation de l'erreur

# je test sur les lignes à partir de 30 000
test=entrainement.copy()
test=test.ix[30000:,]
y=test.ix[:,0]
y=y.as_matrix()
del test['label']
test=test.as_matrix()

# je parcours tous les exemples de test
# je fais un feedforward et regarde la proba la plus élevée
# si la prévision est bonne j'incrémente good
good=0
for i in range(test.shape[0]):
    prevision=neural_net.feedforward(test[i])
    max_index, max_value = max(enumerate(prevision), key=operator.itemgetter(1))
    if max_index==y[i]:
        good+=1

# calcul du pourcentage de bon résultat
print(good/test.shape[0])
