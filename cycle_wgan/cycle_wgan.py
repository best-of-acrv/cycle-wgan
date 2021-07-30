import acrv_datasets
from datetime import datetime
import json
import os
import pkg_resources
import torch

from .helpers import create_dir
from .utils.datasets import load


class CycleWgan(object):
    DATASETS = ['awa1', 'cub', 'flo', 'sun']

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
            torch.manual_seed(self.gpu_id)
            torch.cuda.manual_seed(self.gpu_id)
        elif not torch.cuda.is_available():
            raise RuntimeWarning('PyTorch could not find CUDA, using CPU ...')
        else:
            raise RuntimeWarning(
                'PyTorch is using CPU as requested by cpu flag.')

        # Load the model
        # TODO

    def evaluate(self, *, output_directory='./eval_output'):
        pass

    def predict(self, *, image=None, image_file=None, output_file=None):
        pass

    def train(self, *, output_directory='./train_output', train_gan=True):
        # Load in the dataset
        dataset, knn = _load_dataset(self.config['dataset'],
                                     self.config.get('data_dir', None))

        # Create a unique working directory for the output
        root = os.path.join(output_directory,
                            datetime.now().strftime(r'%Y%m%d_%H%M%S'))
        create_dir(root)

        # Perform GAN training if requested
        if train_gan:
            gan = 0


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
