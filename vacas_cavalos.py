# -*- coding: utf-8 -*-
"""
@author: André Alves
"""


#Bibliotecas utilizadas
import emoji
import numpy as np

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


#Definição da entrada: camadas de convolução
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization()) 
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())


#Camadas ocultas da rede neural densa
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

# Camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])


#' Normalizaão e "Augumentation"
gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.3,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.3)

gerador_teste = ImageDataGenerator(rescale =  1./255)

#Criando a base de treinamento
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
#Criando a base de teste
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

#treinamento
classificador.fit_generator(base_treinamento, steps_per_epoch = 2000 / 32,
                            epochs = 20, validation_data = base_teste,
                            validation_steps = 600 / 32)


#base_treinamento.class_indices

#Classificador: Estou passando o caminho da imagem que eu quero classificar
imagem_teste = image.load_img('dataset/pratica/cavalos/01.jpeg',
                              target_size = (64,64))
#Conversão da imagem
imagem_teste = image.img_to_array(imagem_teste)
#Normalização da imagem
imagem_teste /= 255

imagem_teste = np.expand_dims(imagem_teste, axis = 0)

#Classificando a imagem
print(" \n ---------------------------------------------------------------")

previsao = classificador.predict(imagem_teste)

if (previsao <= 0.4):
    print(emoji.emojize('CAVALO :horse:', use_aliases=True))
    
elif(previsao >= 0.6):
    print(emoji.emojize('VACA :cow:', use_aliases=True))
    
else:
    print("\n NÃO RECONHECIDO")
    
    
    
    

#*REFERÊNCIAS*

#https://keras.io/visualization/
#http://deeplearningbook.com.br/reconhecimento-de-imagens-com-redes-neurais-convolucionais-em-python-parte-1/
#http://conteudo.icmc.usp.br/pessoas/moacir/papers/Ponti_Costa_Como-funciona-o-Deep-Learning_2017.pdf



