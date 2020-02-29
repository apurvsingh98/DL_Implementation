import tensorflow as tf  #importing tensorflow
from keras.datasets import mnist # importing mnist dataset
from keras import models  #importing models which will have our sequential model
from keras import layers  #this module provides different functionality like droupout and dense

(train_images , train_labels),(test_images , test_labels) = mnist.load_data() #importing mnist dataset into train and test images and labels

network = models.Sequential()                          # the standard sequential model used for designing the standard NN which is basically a stack of layers
network.add(layers.Dense(120, activation='relu', input_shape=(28 * 28,)))   # dense is used to implement the "activation(dot(input, kernel) + bias)"
network.add(layers.Dense(95, activation='relu'))
network.add(layers.Dropout(0.3)) 
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
				


train_images = train_images.reshape((60000, 28 * 28))    # selecting the number of images as well as choosing the input size 
train_images = train_images.astype('float32') / 255      # normalizing the image to reduce the computational complexity
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=50, batch_size=128)   

test_loss, test_acc = network.evaluate(test_images, test_labels)  
print('test_acc:', test_acc, 'test_loss', test_loss)	

import matplotlib.pyplot as plt

import matplotlib
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'])
plt.show()
