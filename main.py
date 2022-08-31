# install dependencies
import tensorflow as tf
import keras
import time
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot
import scipy.special
%matplotlib inline

from keras.utils import np_utils
from keras.datasets import mnist
import seaborn as sns
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

# download dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
n_train = x_train.shape[0]
n_test = x_test.shape[0]

x_train = x_train / 127.5 - 1
x_test =  x_test / 127.5 - 1 

nb_features = np.prod(x_train.shape[1:])
x_train.resize((n_train, nb_features))
x_test.resize((n_test, nb_features))

x_train.shape
x_test.shape

# Principal component analysis for 2 components
pca = PCA(n_components=2)
projected = pca.fit_transform(x_train)
print(projected.shape)

# Plot of PCA
plt.scatter(projected[:, 0], projected[:, 1],
            c=y_train, edgecolor='none', alpha=1,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('z1')
plt.ylabel('z2')
plt.colorbar()

# Creation of binary data sets for the values 0 and 1
cond = (y_train == 0) + (y_train == 1)

binary_x_train = x_train[cond,:]
binary_y_train = y_train[cond]*1.

binary_y_train[binary_y_train == 0] = -1
binary_y_train[binary_y_train == 1] = 1

# PCA for these two digits
pca = PCA(2)
projected = pca.fit_transform(binary_x_train)

plt.scatter(projected[:, 0], projected[:, 1],
            c=binary_y_train, edgecolor='none', alpha=1,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

# Single Layer Perceptron implementation

# step function that returns 1 if the input value is greater than 0 and -1 if it is less than.
def step(z):
    return 1 if z >= 0 else -1

# predict function that calculates the dot product of 
# the input values with the weights plus the bias term
def predict(x,w,b):
    z = np.dot(w, x) + b
    return step(z)

# optimise function that will iterate until the iteration limit is reached or the error is 
# of a small enough value. 
def optimise(x,y, iteration_val=1000):
    iteration = 0
    error = np.inf
    n, m = x.shape
    # weights and bias term initialised 
    w = np.random.rand(m) # a weight for each input value 
    b = np.random.rand()
    error_track = []
    while (iteration <= iteration_val) & (error > 1e-3):
        error = 0
        for idx, x_i in enumerate(x): # iterate through each input of x in the training data
            y_hat = predict(w,x_i,b) # the prediction of a certain input 
            error += abs(y[idx] - y_hat) # the absolute error of the input with the t
            update = 0.001 * (y[idx] - y_hat) # multiplication of the update term by the learning rate
            w += update * x_i # upate the weights 
            b += update # update the bias term
        error_track.append(error)
        iteration += 1
    return  w, b, error_track
    
    
# plot of perceptron implementation for 0 and 1 of the error vs the epoch until convergence
w,b, error_track = optimise(binary_x_train,binary_y_train)
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Error')
x = list(range(len(error_track)))
vy =  error_track
ax.plot(x, vy, 'b')
plt.legend()
plt.grid()

# Plot of weights of single layer perceptron after 1000 epochs
image_array = np.asfarray(w).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

# comparison of pairs of values in the single layer perceptron
# plot of PCA, the error vs epoch and the weights 
final_error = []
for val_1, val_2 in zip([0,2,4,6,8], [9,9,9,9,9]):
    cond = (y_train == val_1) + (y_train == val_2)

    binary_x_train = x_train[cond,:]
    binary_y_train = y_train[cond]*1.

    binary_y_train[binary_y_train == val_1] = -1
    binary_y_train[binary_y_train == val_2] = 1
    
    pca = PCA(2)
    projected = pca.fit_transform(binary_x_train)
    plt.scatter(projected[:, 0], projected[:, 1],
            c=binary_y_train, edgecolor='none', alpha=1,
            cmap=plt.cm.get_cmap('nipy_spectral', 10), label=[val_1,val_2])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

    w,b, error_track = optimise(binary_x_train,binary_y_train, iteration_val=50)
    final_error.append(error_track[-1])
    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Error')
    x = list(range(len(error_track)))
    vy = error_track
    plt.plot(x, vy, 'b', label=[val_1,val_2])
    plt.ylim(bottom=-50, top=3000)
    plt.legend()
    plt.grid()
    plt.show()
    
    image_array = np.asfarray(w).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show() 
    
print(final_error)
    
    
# one hot encoding of target values
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# MLP architecture 
input_layer = x_train.shape[1]
hidden_layer_one = 1000
hidden_layer_two = 1000
output_layer = 10

batch_size = 50
learning_rate = 0.001
n_epoch = 10

# creation of MLP model
MLP_model = Sequential()
MLP_model.add(Dense(1000, activation='relu', input_shape=(input_layer,)))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(output_layer, activation='softmax'))

MLP_model.summary()
MLP_model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

# training of the MLP model
history = MLP_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(x_test, y_test))

score = MLP_model.evaluate(x_test, y_test) 
print(score)
print('Test score:', score[0]) 
print('Test accuracy:', score[1])

# plot the loss vs epoch for the MLP model
def plot_epoch_history(epoch, history):
    fig,ax = plt.subplots(1,1)
    ax.set_xlabel('epoch') ; ax.set_ylabel('Accuracy Rate')
    x = list(range(1,n_epoch+1))
    vy = history.history['val_loss']
    ty = history.history['loss']
    ax.plot(x, vy, 'b', label='Validation Loss')
    ax.plot(x, ty, 'r', label='Train Loss')
    plt.legend()
    plt.grid()
    
# Creation of MLP model with different architecture
MLP_model = Sequential()
MLP_model.add(Dense(1000, activation='relu', input_shape=(input_layer,)))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(100, activation='relu'))
MLP_model.add(Dense(output_layer, activation='softmax'))

MLP_model.summary()

MLP_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = MLP_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))

plot_epoch_history(n_epoch, history)

MLP_model = Sequential()
MLP_model.add(Dense(1000, activation='relu', input_shape=(input_layer,)))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(100, activation='relu'))
MLP_model.add(Dense(100, activation='relu'))
MLP_model.add(Dense(output_layer, activation='softmax'))

MLP_model.summary()

MLP_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = MLP_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))
plot_epoch_history(n_epoch, history)

MLP_model = Sequential()
MLP_model.add(Dense(500, activation='relu', input_shape=(input_layer,)))
MLP_model.add(Dense(100, activation='relu'))
MLP_model.add(Dense(100, activation='relu'))
MLP_model.add(Dense(28, activation='relu'))
MLP_model.add(Dense(output_layer, activation='softmax'))

MLP_model.summary()

MLP_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = MLP_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))
plot_epoch_history(n_epoch, history)


MLP_model = Sequential()
MLP_model.add(Dense(1000, activation='relu', input_shape=(input_layer,)))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(output_layer, activation='softmax'))

MLP_model.summary()

MLP_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = MLP_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))
plot_epoch_history(n_epoch, history)


MLP_model = Sequential()
MLP_model.add(Dense(1000, activation='relu', input_shape=(input_layer,)))
MLP_model.add(Dense(1000, activation='relu'))
MLP_model.add(Dense(500, activation='relu'))
MLP_model.add(Dense(100, activation='relu'))
MLP_model.add(Dense(100, activation='relu'))
MLP_model.add(Dense(10, activation='relu'))
MLP_model.add(Dense(output_layer, activation='softmax'))

MLP_model.summary()

MLP_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = MLP_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))
plot_epoch_history(n_epoch, history)


# download of the dataset and into the correct format
mnist = tf.keras.datasets.mnist

# standardizing the data from pixels to be between 0 and 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32')/255
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

# creation of CNN model
CNN_model1 = Sequential()
CNN_model1.add(Conv2D(32, kernel_size=4, strides=(1, 1), activation='relu', input_shape=(28,28,1)))
CNN_model1.add(Conv2D(64, kernel_size=4, strides=(2, 2),activation='relu'))
CNN_model1.add(Conv2D(128, kernel_size=4, strides=(2, 2), activation='relu'))
CNN_model1.add(Flatten())
CNN_model1.add(Dense(10, activation='softmax'))

CNN_model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = CNN_model1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epoch)

CNN_model1.summary()
plot_epoch_history(n_epoch, history)


CNN_model = Sequential()
CNN_model.add(Conv2D(32, kernel_size=4, strides=(1, 1), activation='relu', input_shape=(28,28,1)))
CNN_model.add(Conv2D(64, kernel_size=4, strides=(2, 2),activation='relu'))
CNN_model.add(Flatten())
CNN_model.add(Dense(10, activation='softmax'))

CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = CNN_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epoch)

plot_epoch_history(n_epoch, history)


CNN_model = Sequential()
CNN_model.add(Conv2D(32, kernel_size=4, strides=(1, 1), activation='relu', input_shape=(28,28,1)))
CNN_model.add(Conv2D(64, kernel_size=4, strides=(2, 2),activation='relu'))
CNN_model.add(Conv2D(128, kernel_size=4, strides=(2, 2), activation='relu'))
CNN_model.add(Conv2D(128, kernel_size=4, strides=(2, 2), activation='relu'))
CNN_model.add(Flatten())
CNN_model.add(Dense(10, activation='softmax'))

CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = CNN_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epoch)

plot_epoch_history(n_epoch, history)


CNN_model = Sequential()
CNN_model.add(Conv2D(128, kernel_size=4, strides=(1, 1), activation='relu', input_shape=(28,28,1)))
CNN_model.add(Conv2D(64, kernel_size=4, strides=(2, 2),activation='relu'))
CNN_model.add(Conv2D(32, kernel_size=4, strides=(2, 2), activation='relu'))
CNN_model.add(Flatten())
CNN_model.add(Dense(10, activation='softmax'))

CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = CNN_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epoch)

plot_epoch_history(n_epoch, history)


CNN_model = Sequential()
CNN_model.add(Conv2D(32, kernel_size=4, strides=(1, 1), activation='relu', input_shape=(28,28,1)))
CNN_model.add(Conv2D(64, kernel_size=4, strides=(5, 5),activation='relu'))
CNN_model.add(Conv2D(128, kernel_size=4, strides=(5, 5), activation='relu'))
CNN_model.add(Flatten())
CNN_model.add(Dense(10, activation='softmax'))


CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = CNN_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epoch)
plot_epoch_history(n_epoch, history)


model = CNN_model1

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# visualisation of activation layers
for i in [1, 7]: # it is know that the output at index 1 and 7 contain the wanted values of 2 and 9
    x_i = np.expand_dims(x_test[i], axis=(0)) # reset the dimensions of x
    activations = activation_model.predict(x_i) # and generate activations

    layer_outputs = [layer.output for layer in model.layers ] # find the outputs at each layer of the model
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # find the activations at each layer

    layer_names = []
    for layer in model.layers[:3]: # the first 3 layers are the convolutional layers
        layer_names.append(layer.name)

    # output the activations in a nice format for each layer
    images_per_row = 16 
    for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                 for row in range(images_per_row):
                        channel_image = layer_activation[0,:, :, col * images_per_row + row]
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size : (col + 1) * size,  row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

# functions taken from keras online and applied to the dataset to visualze filters in the first layer
import numpy as np
import tensorflow as tf
from tensorflow import keras

img_width = 28
img_height = 28
layer_name = 'conv2d' # the layer to visualise filters of, this is the first 

layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    filter_activation = activation[:, :, :, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    grads = tape.gradient(loss, img)
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image():
    img = tf.random.uniform((1, img_width, img_height, 1))
    return (img - 0.5) * 0.25

def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 1))
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype("uint8")
    return x

def visualize_filter(filter_index):
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
    img = deprocess_image(img.numpy())
    return loss, img

from IPython.display import Image, display
for i in range(0,32): # there are 32 filters in the first layer
    loss, img = visualize_filter(i)
#     img = tf.keras.preprocessing.image.smart_resize(img,size, interpolation='bilinear')
    keras.preprocessing.image.save_img("0.png", img)
    display(Image("0.png"))
    
import keras.datasets.fashion_mnist as fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_data():
    (train_x, train_y_1), (test_x, test_y_1) = fashion_mnist.load_data()
    n_class_1 = 10
    
    train_y_2 = list(0 if y in [5,7,9] else 1 if y in [3,6,8] else 2 for y in train_y_1)
    test_y_2 = list(0 if y in [5,7,9] else 1 if y in [3,6,8] else 2 for y in test_y_1)
    n_class_2 = 3
    
    # inputs need to be scaled to be inbetween the values of 0 and 1
    train_x = train_x/255
    test_x = test_x/255
    
    train_x = np.expand_dims(train_x, axis=3)
    test_x = np.expand_dims(test_x, axis=3)
    train_y_1 = to_categorical(train_y_1, n_class_1)
    test_y_1 = to_categorical(test_y_1, n_class_1)
    train_y_2 = to_categorical(train_y_2, n_class_2)
    test_y_2 = to_categorical(test_y_2, n_class_2)
    
    return train_x, train_y_1, train_y_2, test_x, test_y_1, test_y_2

x_train, y_train_1, y_train_2, x_test, y_test_1, y_test_2 = load_data()


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input

model = Sequential()
model.add(Conv2D(32, kernel_size=3, strides=(1, 1), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=3, strides=(1, 1),activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense((3136), activation='relu'))
model.add(Dense((1024), activation='relu'))
model.add(Dense((100), activation='relu'))
model.add(Dense((10), activation='softmax'))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 10
n_epoch = 5

history = model.fit(x_train, y_train_1, validation_data=(x_test, y_test_1), epochs=n_epoch)

def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label='Validation Loss')
    ax.plot(x, ty, 'r', label='Train Loss')
    plt.legend()
    plt.grid()
    fig.canvas.draw()
    
    
score = model.evaluate(x_test, y_test_1, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
x = list(range(1,n_epoch+1))
vy = history.history['val_loss']
ty = history.history['loss']
plt.ylim(bottom=0, top=0.4)
plt_dynamic(x, vy, ty, ax)

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras

model = Sequential()
model.add(Conv2D(32, kernel_size=3, strides=(1, 1), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=3, strides=(1, 1),activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, kernel_size=3, strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense((3136), activation='relu'))
model.add(Dense((1024), activation='relu'))
model.add(Dense((100), activation='relu'))
model.add(Dense((3), activation='softmax'))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train_2, validation_data=(x_test, y_test_2), epochs=n_epoch)


score = model.evaluate(x_test, y_test_2, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)
ax.set_ylabel('Categorical Crossentropy Loss')
x = list(range(1,n_epoch+1))
vy = history.history['val_loss']
ty = history.history['loss']
plt.ylim(bottom=0, top=0.4)
plt_dynamic(x, vy, ty, ax)



from tensorflow.keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# creation of shared layers
input1=Input(shape=(28,28,1))
conv1=Conv2D(32, kernel_size=3, strides=(1, 1), activation='relu')
maxpool1=MaxPooling2D(pool_size=(2,2),strides=2)
conv2=Conv2D(64, kernel_size=3, strides=(1, 1), activation='relu')
maxpool2=MaxPooling2D(pool_size=(2,2),strides=2)
conv3=Conv2D(128, kernel_size=3, strides=(1, 1), activation='relu')
dense1= Dense(3136, activation='relu')

model=conv1(input1)
model=maxpool1(model)
model=conv2(model)
model=maxpool2(model)
model=conv3(model)
model=Flatten()(model)
model=dense1(model)

#creation of task specific layers 
task1=Dense(1024, activation='relu')(model)
task1=Dense(100, activation='relu')(task1)
task1=Dense(10, activation='softmax', name='output1')(task1)

task2=Dense(1024, activation='relu')(model)
task2=Dense(100, activation='relu')(task2)
task2=Dense(3, activation='softmax', name='output2')(task2)


model = Model(inputs=[input1], outputs=[task1, task2])

# create loss functions for each model
def task1_loss_funct(y_test_1, y_pred_1):
    task1_loss = K.categorical_crossentropy(y_test_1, y_pred_1)
    return val * task1_loss

def task2_loss_funct(y_test_2, y_pred_2):
    task2_loss = K.categorical_crossentropy(y_test_2, y_pred_2)
    return (1- val) * task2_loss

# loss = val * task1_loss + (1- val) * task2_loss

#iterate through the loss functions
for val in [0, 0.5, 1]:
    opt = Adam(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss={'output1': task1_loss_funct, 'output2': task2_loss_funct} , metrics=['accuracy'])
    history = model.fit(x_train, [y_train_1, y_train_2] ,batch_size=10,validation_data=(x_test,[y_test_1, y_test_2]),epochs=5)

    n_epoch = 5
    x = list(range(1,n_epoch+1))
    vy = history.history['val_loss']
    vz = history.history['val_output1_loss']
    vw = history.history['val_output2_loss']
    ty = history.history['loss']
    tz = history.history['output1_loss']
    tw = history.history['output2_loss']
    plt.plot(x, vy, 'b', label='Validation Loss')
    plt.plot(x, ty, 'r', label='Train Loss')
    plt.legend()
    plt.ylim(bottom=-0.4, top=0.4)
    plt.grid()
    plt.show()
    plt.plot(x, vz, 'b', label='Validation Loss Output1')
    plt.plot(x, tz, 'r', label='Train Loss Output1')
    plt.legend()
    plt.ylim(bottom=-0.3, top=0.3)
    plt.grid()
    plt.show()
    plt.plot(x, vw, 'b', label='Validation Loss Output2')
    plt.plot(x, tw, 'r', label='Train Loss Output2')
    plt.legend()
    plt.ylim(bottom=-0.3, top=0.3)
    plt.grid()
    plt.show()





