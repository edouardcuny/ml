#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

##############################################################################

class NeuralNetwork:

#_____________________________________________________________________________
    def __init__(self, sizes,weights=None):
        '''
        Constructeur de mon réseau de neurones.
        Il prend en entrée une liste de tailles qui correspond à l'architecture
        du réseau.
        Le premier élément de cette liste est de la taille de mon input et mon
        dernier élément est de la taille de mon output.
        '''

        self.num_layers=len(sizes) # nombre de couches de mon réseau
        self.sizes=sizes

        # les poids et biais de mon réseau de neurones sont initialisés de façon
        # aléatoire en suivant la loi normale de paramètres 0 et 1
        self.biases=[np.random.randn(x,1) for x in sizes[1:]]

        if(weights==None):
            self.weights=[np.random.randn(x,y) for y,x in zip(sizes[:-1],sizes[1:])]

        # dans l'éventualité où je veux créer un réseau de neurones avec des
        # poids spécifiques je peux spécifier ces poids

        else:
            self.weights=weights
#_____________________________________________________________________________

    def feedforward(self,x):
        '''
        Feedforward
        Input = un vecteur x, une observation
        Output = l'output de mon réseau de neurones
        '''

        a=x.copy()
        a.shape=(a.size,1) # on transpose le vecteur

        # pour chaque layer on calcule z puis on met à jour a
        for j in range(self.num_layers-1):
                # calcul de z
                produit=np.dot(self.weights[j],a)
                z=np.add(produit,self.biases[j])

                # calcul de a
                a=sigmoid(z) # calcul de a

        return (a)

#_____________________________________________________________________________

    def SGD(self,X,batch_size,epochs):

        '''
        STOCHASTIC GRADIENT DESCENT

        J'ai un vecteur X avec mes features et mes labels
        Mes features sont les colonnes et j'ai un exemple par ligne
        Les dernières colonnes correspondent aux labels

        J'ai un BATCH_SIZE qui dit sur cb d'éléments par iteration je fais ma
        descente de gradient

        J'ai un nombre d'EPOCHS qui me dit cb de fois je parcours mes données
        '''

        # je récupère la taille de l'output pour subset X
        output_size=self.sizes[-1]
        # print(output_size)

        number_observations=X.shape[0] # nombre d'exemples

        # je calcule le nombre cb de backprop je vais faire pour une epoch
        # ça correspond à (nb données) / (taille batch)
        iterations=int(number_observations/batch_size)


        for e in range(epochs):
            print("epoch #",e)

            # je mélange X parce qu'on fait une STOCHASTIC GD
            np.random.shuffle(X)
            for p in range(iterations):
            # pour chaque tour je fais un backprop et mets à jour mes
            # poids et biais

                x=X[(batch_size*p):(batch_size*(p+1)),:-output_size]
                y=X[(batch_size*p):(batch_size*(p+1)),-output_size:]
                self.backprop(x,y)

#_____________________________________________________________________________

    def backprop(self,x,Y,alpha=3):
        x=np.transpose(x)
        Y=np.transpose(Y)

        '''
        / ! \ Mes features sont mes colonnes et j'ai un exemple par ligne

        j'ai un réseau de neurones : SELF
        un ensemble de training data : X
        et des training labels : Y
        ALPHA est mon learning rate

        output : le gradient des biais / le gradient des poids
        c'est une méthode qui change les poids de self

        l'idée est de :
            pour chaque training example
            je calcule tous les z et les a de façon à calculer les DELTA (feedforward)
            à partir des DELTA je peux calculer le GRADIENT
            et ainsi mettre à jour la valeur des POIDS et BIAIS

            les POIDS et BIAIS ainsi obtenus seront la contribution des
            modifications engendrées par tous les exemples

        NB : les calculs que j'éfectue s'inspirent des résultats de M.Nielsen
        dans son livre online

        '''
        # contribution totale = somme(contribution de chq exemple)
        # on parcourt donc tous les exemples
        for i in range(len(x[0])):

            a=x[:,i]
            a.shape=(a.size,1) # reshape pr faciliter calculs matriciels
            y=Y[:,i]
            y.shape=(y.size,1) # reshape pr faciliter calculs matriciels

            A=[] # liste de toutes mes couches d'activation
            Z=[] # idem

            A.append(a)
            Z.append(a)

            delta_list=[] # je mettrais tous mes delta là dedans

            # FEEDFORWARD
            for j in range(self.num_layers-1): # on parcourt toutes les layers
                produit=np.dot(self.weights[j],a) # z avant l'addition des biais
                z=np.add(produit,self.biases[j]) # calcul de z
                a=sigmoid(z) # calcul de a
                A.append(a) # je rajoute a à ma liste
                Z.append(z) # je rajoute z à ma liste

            # CALCUL DE DELTA_L - qui est le delta de la dernière layer
            gradient_C=np.add(A[self.num_layers-1],-y)
            delta=gradient_C*sigmoid_prime(Z[self.num_layers-1])
            delta_list.append(delta)

            # CALCUL DES AUTRES DELTA
            for j in range(self.num_layers-2,0,-1):


                #print("\n\n delta \n",delta)
                #print("\n\n weights \n",self.weights[j])
                #print("\n\n sigmo \n",sigmoid_prime(Z[j]))
                #print("\n\n z \n",Z[j])


                delta=np.dot(np.transpose(self.weights[j]),delta)*sigmoid_prime(Z[j])
                delta_list.append(delta)


            A=A[::-1]

            # CALCUL DU GRADIENT POUR LES POIDS
            gradient_weights=[]
            for j in range(0,len(self.weights)):
                # print("delta",delta_list[j])
                # print("A",np.transpose(A[j+1]))
                gradient=np.dot(delta_list[j],np.transpose(A[j+1]))
                gradient_weights.append(gradient)

            gradient_weights=gradient_weights[::-1]
            delta_list=delta_list[::-1]

            # CALCUL DU GRADIENT POUR LES BIAIS
            gradient_biases=delta_list

            # MAJ POIDS ET BIAIS POUR LE NN
            for j in range(len(self.weights)):
                # print(" \n\n nan ? ",gradient_weights[j])
                self.weights[j]=np.add(self.weights[j],-alpha*gradient_weights[j])
            for j in range(len(self.biases)):
                self.biases[j]=np.add(self.biases[j],-alpha*gradient_biases[j])

            '''
            A ce moment là tout est bon j'ai mon réseau de neurones qui est mis à jour pour ces valeurs.
            '''

    def gradient_checking(self,epsilon,x,y):

        '''
        L'algorithme de backprop est assez difficile à implémenter.
        Gradient Checking est une façon de vérifier que Backprop calcule
        bien le gradient.

        J'ai x, un vecteur ligne correspondant à une observation.
        y les labels associés.

        Je finis par printer le gradient calculé avec backprop et avec gradient
        checking et je vérifie à la main si c'est le même.
        '''

        # gradient unrolled obtenu par le calcul
        gradient_unrolled_calculus=[]

        # UNROLL DES WEIGHTS
        # self.weights est une liste d'arrays correspondant aux poids de
        # mon réseau - le procédé "d'unroll" consiste à transformer cette
        # liste d'array (de matrices) en une liste tout court
        weights_unrolled=self.weights[0].ravel()
        for i in range(1,len(self.weights)):
            ur=self.weights[i].ravel()
            weights_unrolled=np.append(weights_unrolled,ur)


        # pour chaque poids je crée deux réseaux de neurones
        # le premier a les mêmes poids que self à ceci prêt qu'on a rajouté
        # epsilon au ième poids
        # même principe pour le deuxième avec un moins epsilon
        for k in range(len(weights_unrolled)):

            # RESHAPE
            # pour créer mes deux nouveaux réseaux de neurones je dois
            # redonner le bon format aux poids qui sont maintenant en liste
            # c'est le reshape

            # RESEAU +
            eps_plus=weights_unrolled.copy()
            eps_plus[k]+=epsilon
            weights_plus=[]
            idx=0
            for i in range(len(self.weights)):
                shape=self.weights[i].shape
                size=self.weights[i].size
                candidat=eps_plus[idx:idx+size,]
                candidat.shape=shape
                weights_plus.append(candidat)
                idx+=size
            nn_plus=NeuralNetwork(self.sizes,weights_plus)
            nn_plus.biases=self.biases

            # RESEAU -
            eps_minus=weights_unrolled.copy()
            eps_minus[k]-=epsilon
            weights_minus=[]
            idx=0
            for i in range(len(self.weights)):
                shape=self.weights[i].shape
                size=self.weights[i].size
                candidat=eps_minus[idx:idx+size,]
                candidat.shape=shape
                weights_minus.append(candidat)
                idx+=size
            nn_minus=NeuralNetwork(self.sizes,weights_minus)
            nn_minus.biases=self.biases

            # maintenant que j'ai mes deux réseaux de neurons je n'ai plus qu'à
            # effectuer le calcul
            gradient=(nn_plus.cost(x,y)-nn_minus.cost(x,y))/(2*epsilon)
            gradient_unrolled_calculus.append(gradient)

        # j'unroll mon gradient backprop pour faciliter la comparaison avec
        # le gradient calculus
        bp_biases,bp_weights=self.backprop(x,y)

        gradient_unrolled_bp=bp_weights[0].ravel()
        for i in range(1,len(bp_weights)):
            ur=bp_weights[i].ravel()
            gradient_unrolled_bp=np.append(gradient_unrolled_bp,ur)

        print("calculus",gradient_unrolled_calculus)
        print("bp",gradient_unrolled_bp.tolist())
        print([x-y for x,y in zip(gradient_unrolled_calculus,gradient_unrolled_bp.tolist())])


    def cost(self,x,y):
        '''
        Coût QUADRATIQUE pour une observation x associée à un label y

        x est un vecteur - np array (n,1)
        y est un vecteur - np array (m,1)
        calcule le coût pour un vecteur
        '''

        output=self.feedforward(x)
        cost=0.5*np.sum((np.add(output,-y)**2))
        return(cost)



##############################################################################

def sigmoid(x):
    # fonction sigmoide, qui est la fonction d'activation de mon réseau de neurones
    return(1.0/(1.0+np.exp(-x)))

def sigmoid_prime(x):
    # dérivée de la fonction sigmoide
    return((np.exp(-x))/((1.0+np.exp(-x))**2))
