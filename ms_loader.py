
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
        self.image_width = 224
        self.image_height = 224
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
                second_positive_sample_index = random.randint(0, number_of_positive_samples-1)
                # if its the same image, choose another
                #if second_positive_sample_index == current_sample_index:
                #    second_positive_sample_index += 1
                # import pdb
                # pdb.set_trace()
                second_positive_sample_path = os.path.join(self.dataset_path, 'training', current_category,
                                                           available_positive_samples[second_positive_sample_index])
                image = self._load_image(second_positive_sample_path)
                pairs[1][i, :, :, :] = image
            else : 
                negative_category_index = random.randint(0, len(self._train_categories) - 1)
                # make the the category is different
                while negative_category_index == self._current_train_category_index:
                    negative_category_index =random.randint(0, len(self._train_categories) - 1)
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

    def get_one_shot_batch(self, support_set_size=-1):
        ''' Returns One Shot task images. 
        The selected category image will be compared with a support set of images.
        The first pair is always labeled 1, and the remaning ones are 0's. 
        Implementing N way learning only. N being total number of categories.'''
        
        all_categories = self._evaluation_categories
        current_category_index = self._current_evaluation_category_index
        current_category = self._evaluation_categories[current_category_index]
        number_of_categories = len(all_categories)
        if support_set_size == -1:
            support_set_size = number_of_categories

        # pair = [test_sample, second_samples]
        pairs = [np.zeros((support_set_size, self.image_height, self.image_height, 3))
                 for i in range(2)]
        labels = np.zeros((support_set_size,))

        # first pair will always be a set of positive sample images
        labels[0] = 1
        
        # test image, 1st sample image in the pair
        current_category_path = os.path.join(self.dataset_path, 'validation', current_category)
        available_test_samples = os.listdir(current_category_path)
        number_of_available_test_samples = len(available_test_samples)
        test_sample_index = random.randint(0, number_of_available_test_samples-1)
        test_sample_path = os.path.join(current_category_path, available_test_samples[test_sample_index])
        test_image = self._load_image(test_sample_path)

        pairs[0][:, :, :, :] = test_image

        # 2nd sample image in the pair, 1st index will be positive sample
        positive_sample_index = random.randint(0, number_of_available_test_samples-1)
        positive_sample_path = os.path.join(current_category_path,
                                            available_test_samples[positive_sample_index])
        positive_image = self._load_image(positive_sample_path)

        pairs[1][0, :, :, :] = positive_image

        # 2nd sample image for rest of the batch
        # deleting current category
        # all_categories.pop(current_category_index)
        # iterating over all remaining categories and select random image from the available samples
        for i, negative_category in enumerate(all_categories):
            negative_category_path = os.path.join(self.dataset_path, 'validation',
                                                  negative_category)
            available_negative_samples = os.listdir(negative_category_path)
            number_of_available_negative_samples = len(available_negative_samples)
            negative_sample_index = random.randint(0, number_of_available_negative_samples-1)
            negative_sample_path = os.path.join(negative_category_path,
                                                available_negative_samples[negative_sample_index])
            negative_image = self._load_image(negative_sample_path)

            pairs[1][i+1, :, :, :] = negative_image

        return pairs, labels

    def one_shot_test(self, model, number_of_tasks_per_category):
        ''' Prepare one shot task and evaluate.
        returns:
            mean_accuracy: mean accuracy for the one shot task.'''

        mean_global_accuracy = 0

        evaluation_categories = self._evaluation_categories
        
        for category in evaluation_categories:
            mean_category_accuracy = 0
            for _ in range(number_of_tasks_per_category):
                images, _ = self.get_one_shot_batch()
                probabilities = model.predict_on_batch(images)

                # 0th index pair should have maximum probability, all remaining pairs are negatives
                if np.argmax(probabilities) == 0:
                    accuracy = 1.0
                else:
                    accuracy = 0.0

                mean_category_accuracy += accuracy
                mean_global_accuracy += accuracy
            mean_category_accuracy /= number_of_tasks_per_category

            print('Accuracy of category {} = {}'.format(category, str(mean_category_accuracy)))

            self._current_evaluation_category_index += 1
            if self._current_evaluation_category_index >= (len(evaluation_categories) -1):
                self._current_evaluation_category_index = 0

        mean_global_accuracy /= (len(evaluation_categories) * number_of_tasks_per_category)
        print('Mean global accuracy: {}'.format(mean_global_accuracy))

        # reseting counter
        self._current_evaluation_category_index = 0

        return mean_global_accuracy
        
# # test
# dataset_path = '../../datasets/monkey_species/'
# batch_size = 4
# loader = MSLoader(dataset_path, batch_size)
# loader.get_train_batch()
# loader.get_one_shot_batch()
