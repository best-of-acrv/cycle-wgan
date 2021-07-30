import acrv_datasets
from datetime import datetime
import json
import os
import pkg_resources
import torch

from . import models
from . import helpers
from .utils.datasets import augment_dataset, load


class CycleWgan(object):
    AUGMENTATION_METHODS = ['none', 'replace', 'merge']
    DATASETS = ['awa1', 'cub', 'flo', 'sun']
    DOMAINS = ['unseen', 'seen', 'unseen seen']

    def __init__(
        self,
        *,
        config=pkg_resources.resource_filename(__name__, '/configs/awa1.json'),
        cpu=False,
        gpu_id=0,
        model_seed=0,
    ):
        # Apply sanitised arguments
        self.config = config
        self.cpu = cpu
        self.gpu_id = gpu_id
        self.model_seed = model_seed

        # Attempt to load the specified config file
        with open(self.config, 'r') as f:
            self.config = json.load(f)

        # Check config for any glaring errors
        _sanitise_arg(self.config['dataset'], 'dataset', CycleWgan.DATASETS)

        # Try setting up GPU integration
        if not self.cpu and torch.cuda.is_available():
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            torch.manual_seed(self.model_seed)
            torch.cuda.manual_seed(self.model_seed)
        elif not torch.cuda.is_available():
            raise RuntimeWarning('PyTorch could not find CUDA, using CPU ...')
        else:
            raise RuntimeWarning(
                'PyTorch is using CPU as requested by cpu flag.')

        # Load the models
        # TODO

    def evaluate(self, *, output_directory='./eval_output'):
        pass

    def predict(self, *, image=None, image_file=None, output_file=None):
        pass

    def train(self,
              *,
              augmentation_method='none',
              domain='unseen seen',
              generate_fake_data=True,
              number_features=[1200, 300],
              output_directory=None,
              train_gan=True,
              train_gzsl=True):
        # Load in the dataset
        dataset, knn = _load_dataset(self.config['dataset'],
                                     self.config.get('data_dir', None))
        device = torch.device('cpu') if self.cpu else None

        # Sanitise & validate arguments
        aug_method = _sanitise_arg(augmentation_method, 'augmentation_method',
                                   CycleWgan.AUGMENTATION_METHODS)
        domain = _sanitise_arg(domain, 'domain', CycleWgan.DOMAINS)
        if not any([train_gan, generate_fake_data, train_gzsl]):
            raise ValueError("Must select at least one of 'train_gan', "
                             "'generate_fake_data', or 'train_gzsl'")

        # Create a unique working directory for the output if none was
        # explicitly provided
        if output_directory == None:
            output_directory = os.path.join(
                './train_output',
                datetime.now().strftime(r'%Y%m%d_%H%M%S'))
            helpers.create_dir(output_directory)

        # Train GAN if requested
        gan = None
        gan_dir = os.path.join(output_directory, 'gan')
        if train_gan:
            gan = helpers.train_gan(device, gan_dir, self.config['GAN'],
                                    dataset.train)

        # Generate a dataset of fake visual samples if requested
        fake_file = os.path.join(output_directory, 'generated_data', 'data.h5')
        if generate_fake_data:
            if gan is None:
                gan = helpers.setup_model(models.GAN, device, gan_dir,
                                          self.config['GAN'])
            helpers.generate_fake_data(gan, knn, fake_file, domain,
                                       number_features)

        # Apply the selected augmentation method in adding fakes to dataset
        if augmentation_method != CycleWgan.AUGMENTATION_METHODS[0]:
            dataset = augment_dataset(dataset, fake_file, augmentation_method)

        # Train GZSL classifier if requested
        classifier = None
        classifier_dir = os.path.join(output_directory, 'gzsl_classifier')
        if train_gzsl:
            classifier = helpers.train_gzsl_classifier(
                device, classifier_dir, self.config['GZSL_classifier'],
                dataset.train)


def _load_dataset(dataset_name, dataset_dir=None, quiet=False):
    # Print some verbose information
    if not quiet:
        print("\nGETTING DATASET:")
    if dataset_dir is None:
        dataset_dir = acrv_datasets.get_datasets(dataset_name)
    if not quiet:
        print("Using 'data_dir': %s" % dataset_dir)

    # Return dataset and k-nearest neighbours from the dataset_dir
    return load(dataset_dir)


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
