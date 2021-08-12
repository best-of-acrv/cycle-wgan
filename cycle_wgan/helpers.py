import os
from sklearn.model_selection import train_test_split

from . import models
from .utils import loaders


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_fake_data(gan, knn, aug_file, domain, num_features):
    #Set-up directory
    create_dir(os.path.dirname(aug_file))

    #Used trained generator to synthesis fake visual samples - save to aug_file
    gan.generator.generate_dataset(aug_file, knn, domain.split(' '),
                                   num_features)


def get_data_as_dict(x=None, y=None, a=None):
    return {'x': x, 'y': y, 'a': a}


def get_split_datasets(dataset, val_size=0.1):

    #Split data into training/validation (1-val_size/val_size)
    split = train_test_split(dataset.X,
                             dataset.Y,
                             dataset.A.continuous,
                             test_size=val_size,
                             random_state=42)

    #Create dataset objects for both training and validation splits
    train_data = get_data_as_dict(split[0], split[2], split[4])
    train_dataset = loaders.LoaderH5(train_data)
    val_data = get_data_as_dict(split[1], split[3], split[5])
    val_dataset = loaders.LoaderH5(val_data)

    return train_dataset, val_dataset


def harmonic_mean(a, b):
    return 2 * a * b / (a + b)


def setup_model(model, device, model_dir, opts):
    create_dir(model_dir)
    return model(device, model_dir, opts)


def train_gan(device, gan_dir, opts, dataset):

    #Set-up directory and create GAN
    gan = setup_model(models.GAN, device, gan_dir, opts)

    #Get train/val datasets
    train_dataset, val_dataset = get_split_datasets(dataset)

    #Train GAN
    gan.train(train_dataset, val_dataset)

    return gan


def train_gzsl_classifier(device, classifier_dir, opts, dataset):

    #Set-up directory and create GZSL classifier
    gzsl_classifier = setup_model(models.Classifier, device, classifier_dir,
                                  opts)

    #Get train/val datasets
    train_dataset, val_dataset = get_split_datasets(dataset)

    #Train a classifier for GZSL on the provided dataset
    gzsl_classifier.train(train_dataset,
                          val_dataset,
                          batch_size=opts['batch_size'])

    return gzsl_classifier


def test_gzsl_classifier(gzsl_classifier, opts, dataset, knn):

    #Get seen/unseen test data
    test_dataset_seen = get_split_datasets(dataset.seen, val_size=0)[0]
    test_dataset_unseen = get_split_datasets(dataset.unseen, val_size=0)[0]

    #Per-class accuracies
    batch_size = opts['batch_size'] if 'batch_size' in opts else 64
    seen_acc = gzsl_classifier.class_accuracy(test_dataset_seen, batch_size,
                                              knn.openval.ids - 1)
    unseen_acc = gzsl_classifier.class_accuracy(test_dataset_unseen,
                                                batch_size, knn.zsl.ids - 1)

    print("Y(U) = {0:.4f}, Y(S) = {1:.4f}, H = {2:.4f}, ".format(
        unseen_acc, seen_acc, harmonic_mean(seen_acc, unseen_acc)))

    return unseen_acc, seen_acc
