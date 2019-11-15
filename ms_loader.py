
import os
import numpy as np
import random
from PIL import Image
import pdb


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
        self.image_width = 500
        self.image_height = 500
        self.batch_size = batch_size
        assert (batch_size % 2) == 0, 'batch size can not be odd'
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

            self.train_dictionary[category] = os.listdir(species_path)

        # loading evaluation monkey specied categories
        for category in os.listdir(evaluation_path):
            species_path = os.path.join(evaluation_path, category)

            self.evaluation_dictionary[category] = os.listdir(species_path)

        self._train_categories = list(self.train_dictionary.keys())
        self._evaluation_categories = list(self.evaluation_dictionary.keys())

    def _load_image(self, image_path):
        ''' Load the image and normalize. Devide by standard deviation and subtract the mean from the image. '''
        image = Image.open(image_path)
        image = image.resize((self.image_width, self.image_height))
        image = np.asarray(image).astype(np.float64)
        image = image / image.std() - image.mean()
        return image

    def get_train_batch(self):
        ''' Loads and returns a batch of train images.
        The batch will contain images from a single category. We will have a batch size of n
        with n/2 pairs of same category having label 1 and n/2 pairs having different categories
        having label 0. In case the number of images in a category in smaller than n/2, 
        the image could be repeated but paired with different category.

        Returns: 
            pairs, labels '''

        pairs = [np.zeros((self.batch_size, self.image_height, self.image_height, 3))
                 for i in range(2)]
        labels = np.zeros((self.batch_size,))
        labels[:self.batch_size//2] = 1
        
        current_category = self._train_categories[self._current_train_category_index]
        available_positive_samples = list(self.train_dictionary[current_category])
        number_of_positive_samples = len(available_positive_samples)
        # let us select the samples which would be used as first image in the pair
        selected_positive_samples_indexes = [random.randint(0, number_of_positive_samples-1)
                                             for i in range(self.batch_size)]
        # positive samples
        for i, current_sample_index in enumerate(selected_positive_samples_indexes):
            current_sample =  available_positive_samples[current_sample_index]
            current_sample_path = os.path.join(self.dataset_path, 'training', current_category, current_sample)
            image = self._load_image(current_sample_path)
            # 1st image of a pair in  ith index of batch
            pairs[0][i, :, :, :] = image

            # 2nd image of the pair, should be positive if i < batch_size / 2, else negative
            if i < self.batch_size / 2:
                second_positive_sample_index = random.randrange(number_of_positive_samples-1)
                if second_positive_sample_index >= current_sample_index:
                    second_positive_sample_index += 1
                second_positive_sample_path = os.path.join(self.dataset_path, 'training', current_category,
                                                           available_positive_samples[second_positive_sample_index])
                image = self._load_image(second_positive_sample_path)
                pairs[1][i, :, :, :] = image
            else : 
                negative_category_index = random.randrange(len(self._train_categories) - 1)
                if negative_category_index >= self._current_train_category_index:
                    negative_category_index += 1
                negative_category_path = os.path.join(self.dataset_path, 'training', self._train_categories[negative_category_index])
                available_negative_samples = os.listdir(negative_category_path)
                number_of_negative_samples = len(available_negative_samples)
                negative_sample_index = random.randint(0, number_of_negative_samples - 1)
                negative_sample_path = os.path.join(negative_category_path,
                                                    available_negative_samples[negative_sample_index])
                image = self._load_image(negative_sample_path)
                pairs[1][i, :, :, :] = image

        self._current_train_category_index += 1
        if (self._current_train_category_index > (len(self._train_categories)-1)):
            self._current_train_category_index = 0

        return pairs, labels

        
# test
dataset_path = '../../datasets/monkey_species/'
batch_size = 4
loader = MSLoader(dataset_path, batch_size)
loader.get_train_batch()

