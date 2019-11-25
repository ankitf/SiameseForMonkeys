
import os
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D
# from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
import numpy as np

from ms_loader import MSLoader

class SiameseNet:
    '''Builds siamesenet for training.'''

    def __init__(self, dataset_path, learning_rate, batch_size, tensorboard_log_path):
        self.input_shape = (224, 224, 3)
        self.model = []
        self.learning_rate = learning_rate
        self.ms_loader = MSLoader(dataset_path=dataset_path, batch_size=batch_size)
        self._build_siamese_net()

    # def _build_siamese_net(self):
    #     ''' Define architecture for SiameseNet.'''

    #     # build the branch model
    #     # base_model = NASNetMobile(self.input_shape, include_top=False, weights='imagenet')
    #     base_model = MobileNetV2(self.input_shape, include_top=False, weights='imagenet')
    #     x = Flatten()(base_model.output)
    #     x = Dense(4096, activation='sigmoid')(x)
    #     branch_model = Model(base_model.input, x, name='branch_model')
    #     plot_model(branch_model, to_file='branch_model.png')
        
    #     # build the head model
    #     xa_input = Input(shape=branch_model.output_shape[1:])
    #     xb_input = Input(shape=branch_model.output_shape[1:])
    #     x = Lambda(lambda tensors: K.abs(tensors[0] -  tensors[1]))([xa_input, xb_input])
    #     x = Dense(1, activation = 'sigmoid')(x)
    #     head_model = Model([xa_input, xb_input], x)
    #     plot_model(head_model, to_file='head_model.png')
        
    #     # build the siamese net
    #     img_a = Input(self.input_shape)
    #     img_b = Input(self.input_shape)
    #     xa = branch_model(img_a)
    #     xb = branch_model(img_b)
    #     x = head_model([xa, xb])
    #     self.model = Model([img_a, img_b], x)
    #     plot_model(self.model, to_file='siamese_net.png')

    #     optimizer = Adam(self.learning_rate)
        
    #     self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
    #                        optimizer=optimizer)

    def _build_siamese_net(self):
        # Let's define the cnn architecture
        convolutional_net = Sequential()
        convolutional_net.add(Conv2D(filters=64, kernel_size=(10, 10),
                                     activation='relu',
                                     input_shape=self.input_shape,
                                     name='Conv1'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=128, kernel_size=(7, 7),
                                     activation='relu',
                                     name='Conv2'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=128, kernel_size=(4, 4),
                                     activation='relu',
                                     name='Conv3'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=256, kernel_size=(4, 4),
                                     activation='relu',
                                     name='Conv4'))

        convolutional_net.add(Flatten())
        convolutional_net.add(
            Dense(units=4096, activation='sigmoid',
                  name='Dense1'))

        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)

        # L1 distance layer between the two encoded outputs
        # One could use Subtract from Keras, but we want the absolute value
        l1_distance_layer = Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

        # Same class or not prediction
        prediction = Dense(units=1, activation='sigmoid')(l1_distance)
        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        # Define the optimizer and compile the model
        optimizer = Adam(lr=self.learning_rate)

        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                           optimizer=optimizer)

        
    def train_siamese_net(self, number_of_iterations, evaluate_each,
                          model_name):
        ''' Training of Siamese Network.'''

        # Store losses at evaluate_each iterations
        train_losses = np.zeros(shape=(evaluate_each))
        train_accuracies = np.zeros(shape=(evaluate_each))
        count = 0
        early_stop = 0

        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0.0
        validation_accuracy = 0.0

        # Train loop
        for iteration in range(number_of_iterations):
            # train set
            images, labels = self.ms_loader.get_train_batch()

            train_loss, train_accuracy = self.model.train_on_batch(images, labels)

            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy

            # validation set
            count += 1
            print('Itera tion {}/{}: Train Loss: {}, Train Accuracy: {}'.format(
                iteration, number_of_iterations, train_loss, train_accuracy))

            
            
