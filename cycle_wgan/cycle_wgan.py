import json
import os
import pkg_resources
import torch


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

    def train(self):
        pass


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
