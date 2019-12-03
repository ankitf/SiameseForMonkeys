
from siamese_net import SiameseNet


def main():
    dataset_path = './monkey_species'
    learning_rate = 10e-4
    batch_size = 8

    tensorboard_logs_path = './logs/'

    print('Initialising Siamese Network...')
    net = SiameseNet(dataset_path, learning_rate, batch_size, tensorboard_logs_path)

    evaluate_each = 1000
    number_of_train_iterations = 10000

    print('Training Siamese Network...')
    evaluation_accuracy = net.train_siamese_net(number_of_train_iterations, evaluate_each,
                                                'siamese_net')

    print('Final Evaluation Accuracy: {}'.format(str(evaluation_accuracy)))


if __name__ == '__main__':
    main()
