import pkg_resources


class CycleWgan(object):
    DATASETS = ['awa1', 'cub', 'flo', 'sun']

    def __init__(
        self,
        *,
        config=pkg_resources.resource_filename('/configs/awa1.json'),
        cpu=False,
        gpu_id=0,
        model_seed=0,
    ):
        print("Created instance...")

    def evaluate(self,
                 dataset_name,
                 *,
                 dataset_dir=None,
                 output_directory='./eval_output'):
        pass

    def predict(self, *, image=None, image_file=None, output_file=None):
        pass

    def train(self, dataset_name, *, dataset_dir=None):
        pass
