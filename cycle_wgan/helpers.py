import os
import pkg_resources
from sklearn.model_selection import train_test_split
from six.moves import urllib
import shutil

from . import models
from .utils import loaders

_CACHE_LOCATION = '.cache'
_CONFIGS_LOCATION = 'configs'

_PRETRAINED_URLS = {
    'awa1': 'https://cloudstor.aarnet.edu.au/plus/s/2eTH1uXaYlsAqT7/download',
    'cub': 'https://cloudstor.aarnet.edu.au/plus/s/tufEmxTm4fvFkn3/download',
    'flo': 'https://cloudstor.aarnet.edu.au/plus/s/2gcuAkZRbUdq8Lf/download',
    'sun': 'https://cloudstor.aarnet.edu.au/plus/s/jXpqVxJRkrrNZAZ/download',
}


def cache_location():
    return pkg_resources.resource_filename(__name__, _CACHE_LOCATION)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_pretrained(pretrained_name):
    cl = cache_location()
    print("Downloading pretrained model to package cache:\n\t%s" % cl)
    if not os.path.exists(cl):
        os.makedirs(cl)
    cached_fn = os.path.join(cl, '%s' % pretrained_name)
    if not os.path.exists(cached_fn):
        cached_dest = '%s.tgz' % cached_fn
        print("Downloading to %s\n" % cached_dest)
        urllib.request.urlretrieve(_PRETRAINED_URLS[pretrained_name],
                                   cached_dest)
        os.makedirs(cached_fn)
        shutil.unpack_archive(cached_dest, cached_fn, format='gztar')
    return cached_fn


def config_by_name(name, must_exist=True):
    fn = pkg_resources.resource_filename(
        __name__, os.path.join(_CONFIGS_LOCATION, '%s.json' % name))
    if must_exist and not os.path.exists(fn):
        raise ValueError('No config exists with the filepath:\n\t%s' % fn)
    return fn


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
    print("Test size: %f" % val_size)
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
    test_dataset_seen = get_split_datasets(dataset.seen)[0]
    test_dataset_unseen = get_split_datasets(dataset.unseen)[0]

    #Per-class accuracies
    batch_size = opts['batch_size'] if 'batch_size' in opts else 64
    seen_acc = gzsl_classifier.class_accuracy(test_dataset_seen, batch_size,
                                              knn.openval.ids - 1)
    unseen_acc = gzsl_classifier.class_accuracy(test_dataset_unseen,
                                                batch_size, knn.zsl.ids - 1)

    print(
        "Y(unseen) = {0:.4f}, Y(seen) = {1:.4f}, Harm mean = {2:.4f}, ".format(
            unseen_acc, seen_acc, harmonic_mean(seen_acc, unseen_acc)))

    return unseen_acc.item(), seen_acc.item()
