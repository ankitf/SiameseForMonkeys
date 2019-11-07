
import os
import numpy as np


class MSLoader:
    ''' 
    Class for loading Monkey Specied Dataset and preparing one 
    Shot tasks for the Siamese Net.

    Attributes:
        dataset_path: path for Monkey Species dataset
        train_dictionary: dictionary for trainset to load the batch for training.
        evaluation_dictionary: dictionary for validation set. Would be used for evaluation as well.
        image_width: all the images would be resized to this width
        image_height: all the images would ne resized to this height
        batch_size: batch size for training
    '''
    
    def __init__(self, dataset_path, batch_size):
        ''' Class attributes initialization and loading the dataset'''
        self.dataset_path = dataset_path
        self.train_dictionary = {}
        self.evaluation_dictionary = {}
        self.image_width = 224
        self.image_height = 224
        self.batch_size = batch_size
        self._train_categories = []
        self._evaluation_categories = []
        self._current_train_category_index =  0
        self._current_evaluation_category_index = 0

        self.load_dataset()

    def load_dataset(self):
        ''' Loads the categories into dictionaries. '''

        train_path = os.path.join(self.dataset_path, 'training')
        evaluation_path = os.path.join(self.dataset_path, 'validation')

        # loading training monkey species categories
        for category in os.listdir(train_path):
            species_path = os.path.join(train_path, category)

            current_category_dictionary = {}
            
