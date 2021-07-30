import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader

from .utils import initialisers
from .utils.datasets import merge_array, merge_dict
from .utils.storage import DataH5py


class BaseModel(nn.Module):
    """
    A class for creating/training/testing/saving models.
    ...

    Attributes:
    device : torch.device
        Device for model training/testing

    Methods:
    validate(val_loader)
        Evaluate the model on the provided validation data
    train_loop(train_loader)
        Perform one training epoch
    train(train_dataset, val_dataset=None, batch_size=64)
        Train the model and validate/save periodically
    load_checkpoint()
        Load a training checkpoint from save_dir
    """

    def __init__(self, device, save_dir):
        """
        Parameters:
        device : torch.device
            Device for model training/testing.
        save_dir : str
            Directory for reading/writing model checkpoints.
        """

        super(BaseModel, self).__init__()
        self.device = device
        self._save_dir = save_dir
        self._start_epoch = 0

    def validate(self, val_loader):
        """
        Evaluate the model on the provided validation data.

        Parameters:
        val_loader : torch.utils.data.Dataloader
            Dataloader for validation data
        Returns:
         : list
            Validation loss as ['Loss', float]
        """

        loss_running = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_loader):

                outputs = self.forward(data[self._inputs].to(self.device))
                loss = self.loss_func(outputs,
                                      data[self._targets].to(self.device))

                loss_running += loss.item()

        return ["Loss", loss_running / float(len(val_loader))]

    def train_loop(self, train_loader):
        """
        Perform one training epoch.

        Parameters:
        train_loader : torch.utils.data.Dataloader
            Dataloader for training data
        Returns:
         : list
            Training loss as ['Loss', float]
        """

        loss_running = 0.0

        for i, data in enumerate(train_loader):

            self.optimiser.zero_grad()

            outputs = self.forward(data[self._inputs].to(self.device))
            loss = self.loss_func(outputs, data[self._targets].to(self.device))

            loss.backward()
            self.optimiser.step()

            self.lr_scheduler.step()

            loss_running += loss.item()

        return ["Loss", loss_running / float(len(train_loader))]

    def train(self, train_dataset, val_dataset=None, batch_size=64):
        """
        Train the model and validate/save periodically. 

        Parameters:
        train_dataset : util.loaders.LoaderH5
            Dataset class for training data.
        val_dataset : util.loaders.LoaderH5, optional
            Dataset class for validation data.
        batch size : int, optional
            Dataloader for training data.
        """

        #Create data loaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=1)

        print(":: Training {0} from epoch {1} / {2}".format(
            self._model_name, self._start_epoch, self._max_epochs))

        for epoch in range(self._start_epoch, self._max_epochs):

            #Train
            train_results = self.train_loop(train_loader)

            self._print_metrics(epoch, "Train", train_results)

            #Validate
            if val_dataset is not None:
                if ((epoch + 1) % self._val_freq
                        == 0) or (epoch + 1 == self._max_epochs):
                    val_results = self.validate(val_loader)
                    self._print_metrics(epoch, "Val", val_results)

            #Save
            if ((epoch + 1) % self._save_freq == 0) or (epoch + 1
                                                        == self._max_epochs):
                self._save_checkpoint(epoch, train_results[1::2])

    def _save_checkpoint(self, epoch, loss):
        #Save training checkpoint (weights, optimiser, lr scheduler, epoch, loss).

        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'loss': loss,
            }, self._save_dir + "/" + self._model_name.replace(" ", "_") +
            "_checkpoint.pt")

    def load_checkpoint(self):
        """
        Load a training checkpoint from save_dir.
        """

        save_file = self._save_dir + "/" + self._model_name.replace(
            " ", "_") + "_checkpoint.pt"
        if os.path.isfile(save_file):
            print(":: Loading {0} checkpoint from {1}".format(
                self._model_name, save_file))

            checkpoint = torch.load(save_file,
                                    map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler_state_dict'])
            self._start_epoch = checkpoint['epoch']

        else:
            print(
                ":: No checkpoint found for {0}, starting from scratch".format(
                    self._model_name))

    def _print_metrics(self, epoch, mode, values):
        #Print training/validation metrics.

        details_str = "[{0:03d} / {1:03d}] {2} :: {3} :: ".format(
            epoch + 1, self._max_epochs, self._model_name, mode.rjust(5))
        print(details_str +
              " / ".join("{0} = {1:7.4f}".format(*values[i:i + 2])
                         for i in range(0, len(values), 2)))

    def _get_optimiser(self, opts, params):
        #Return a torch.nn.optim Adam optimiser for model training.

        lr = opts['lr']
        weight_decay = opts['wdecay'] if 'wdecay' in opts else 0
        beta1 = opts['adam_beta1'] if 'adam_beta1' in opts else 0.9
        beta2 = opts['adam_beta2'] if 'adam_beta2' in opts else 0.999

        return optim.Adam(params,
                          lr=lr,
                          weight_decay=weight_decay,
                          betas=(beta1, beta2))

    def _get_lr_scheduler(self, opts, optimiser):
        #Return a torch.nn.optim learning rate scheduler.

        decay = opts['lr_decay'] if 'lr_decay' in opts else 0
        lr_lambda = lambda global_step: 1 / (1 + global_step * decay)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimiser,
                                                   lr_lambda=lr_lambda)

        return lr_scheduler

    def _get_training_details(self, opts):
        #Set max_epochs, val_freq and save_freq as class attributes.

        self._max_epochs = opts['epochs']
        self._val_freq = opts['val_freq'] if 'val_freq' in opts else 5
        self._save_freq = opts['save_freq'] if 'save_freq' in opts else 5


class Classifier(BaseModel):
    """
    A generator class for creating a model and generating fake features.
    Does not contain training methods - training is handled by the GAN class.
    ...

    Attributes:
    fc1 : torch.nn.Linear
        Fully connected layer 1
    loss_func : torch.nn.Module
        Loss function for training classifier
    optimiser : torch.optim.Optimizer
        Optimiser to update regressor weights
    lr_scheduler : object
        Learning rate scheduler for optimiser (from torch.nn.optim.lr_scheduler)

    Methods:
    forward(x)
        Get logits from visual features
    class_accuracy(self, dataset, batch_size, class_ids)
        Calculate mean per-class accuracy of dataset
    """

    def __init__(self, device, save_dir, opts):
        """
        Parameters:
        device : torch.device
            Device for model training/testing
        save_dir : str
            Directory for reading/writing model checkpoints
        opts : dict
            Dictionary containing classifier training options. Keys include:
                'x_dim' : int
                    Dimensionality of visual features
                'y_dim' : int
                    Number of classes (dimensionality of classifier logits)
                'epochs': int
                    Maximum number of training epochs
                'lr' : float
                    Base learning rate 
                'lrdecay' : float
                    Learning rate decay factor, optional (default = 0)
                'wdecay' : float
                    Weight decay factor, optional (default = 0)
                'adam_beta1' : float
                    Beta1 parameter for Adam optimiser, optional (default = 0.9)
                'adam_beta2' : float
                    Beta2 parameter for Adam optimiser, optional (default = 0.999)
                'val_freq' : int
                    Frequency of model validation during training (epochs), optional (default = 5)
                'save_freq' : int
                    Frequency of model checkpointing during training (epochs), optional (default = 5)
                'name' : str
                    Model name for saving/print, optional (default = 'classifier')
        """

        super(Classifier, self).__init__(device, save_dir)

        self._inputs = 'x'
        self._targets = 'y'
        self._num_classes = opts['y_dim']
        self._model_name = opts['name'] if 'name' in opts else 'classifier'

        self.fc1 = nn.Linear(opts['x_dim'], opts['y_dim'], bias=True)
        self.to(self.device)

        #Set up for training
        self._get_training_details(opts)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimiser = self._get_optimiser(opts, self.parameters())
        self.lr_scheduler = self._get_lr_scheduler(opts, self.optimiser)
        self.apply(initialisers.init_weights_trunc_norm)

        #Load checkpoint (if one exists)
        self.load_checkpoint()

    def forward(self, x):
        """
        Get logits from visual features.

        Parameters:
        x : torch.Tensor
            Visual features

        Returns:
        c : torch.Tensor
            Logits
        """

        c = F.relu(self.fc1(x))

        return c

    def class_accuracy(self, dataset, batch_size, class_ids):
        """
        Calculate mean per-class accuracy of dataset.

        Parameters:
        dataset : util.loaders.LoaderH5
            Dataset class for accuracy computation
        batch_size : int
            Batch size for data loader
        class_ids : numpy.ndarray
            Class labels to include in accuracy computation

        Returns:
        a : float
            Mean per-class accuracy
        """

        loss_running = 0.0

        correct = torch.zeros(self._num_classes)
        total = torch.zeros(self._num_classes)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1)

        with torch.no_grad():
            for i, data in enumerate(loader):

                pred = self.forward(data[self._inputs].to(
                    self.device)).argmax(dim=1).cpu()

                y = data[self._targets]

                for j in range(y.size(0)):
                    correct[y[j]] += pred[j] == y[j]
                    total[y[j]] += 1

        return (correct[class_ids] /
                (total[class_ids]).type(torch.FloatTensor)).mean()

    def _class_accuracy_accum(self, scores, target, correct=None, total=None):
        #Accumulate per-class accuracy for a batch - called from GAN.validate()

        if correct is None:
            correct = torch.zeros(self._num_classes)
            total = torch.zeros(self._num_classes)

        pred = scores.argmax(dim=1).cpu()

        for j in range(target.size(0)):
            correct[target[j]] += pred[j] == target[j]
            total[target[j]] += 1

        return correct, total


class Regressor(BaseModel):
    """
    A generator class for creating a model and generating fake features.
    Does not contain training methods - training is handled by the GAN class.
    ...

    Attributes:
    fc1 : torch.nn.Linear
        Fully connected layer 1
    loss_func : torch.nn.Module
        Loss function for training regressor
    optimiser : torch.optim.Optimizer
        Optimiser to update regressor weights
    lr_scheduler : object
        Learning rate scheduler for optimiser (from torch.nn.optim.lr_scheduler)

    Methods:
    forward(x)
        Transform from visual space to semantic space
    """

    def __init__(self, device, save_dir, opts):
        """
        Parameters:
        device : torch.device
            Device for model training/testing
        save_dir : str
            Directory for reading/writing model checkpoints
        opts : dict
            Dictionary containing regressor training options. Keys include:
                'x_dim' : int
                    Dimensionality of visual features
                'a_dim' : int
                    Dimensionality of semantic features (attributes)
                'epochs': int
                    Maximum number of training epochs
                'lr' : float
                    Base learning rate 
                'lrdecay' : float
                    Learning rate decay factor, optional (default = 0)
                'wdecay' : float
                    Weight decay factor, optional (default = 0)
                'adam_beta1' : float
                    Beta1 parameter for Adam optimiser, optional (default = 0.9)
                'adam_beta2' : float
                    Beta2 parameter for Adam optimiser, optional (default = 0.999)
                'val_freq' : int
                    Frequency of model validation during training (epochs), optional (default = 5)
                'save_freq' : int
                    Frequency of model checkpointing during training (epochs), optional (default = 5)
                'name' : str
                    Model name for saving/print, optional (default = 'regressor')
        """

        super(Regressor, self).__init__(device, save_dir)

        self._inputs = 'x'
        self._targets = 'a'
        self._model_name = opts['name'] if 'name' in opts else 'regressor'

        self.fc1 = nn.Linear(opts['x_dim'], opts['a_dim'], bias=True)
        self.to(self.device)

        #Set-up for training
        self._get_training_details(opts)
        self.apply(initialisers.init_weights_xavier)
        self.loss_func = nn.MSELoss()
        self.optimiser = self._get_optimiser(opts, self.parameters())
        self.lr_scheduler = self._get_lr_scheduler(opts, self.optimiser)

        #Load checkpoint (if one exists)
        self.load_checkpoint()

    def forward(self, x):
        """
        Transform from visual space to semantic space.

        Parameters:
        x : torch.Tensor
            Visual features

        Returns:
        a : torch.Tensor
            Semantic features (attributes)
        """

        a = self.fc1(x)

        return a


class Generator(BaseModel):
    """
    A generator class for creating a model and generating fake features.
    Does not contain training methods - training is handled by the GAN class.
    ...

    Attributes:
    device : torch.device
        Device for model training/testing
    fc1 : torch.nn.Linear
        Fully connected layer 1
    fc2 : torch.nn.Linear
        Fully connected layer 2
    optimiser : torch.optim.Optimizer
        Optimiser to update generator weights
    lr_scheduler : object
        Learning rate scheduler for optimiser (from torch.nn.optim.lr_scheduler)

    Methods:
    forward(a, z=None)
        Generate visual feature from attribute and noise
    generate_dataset(aug_file, knn, domain = ['unseen'], num_features = [200])
        Generate and save fake visual features
    """

    def __init__(self, device, save_dir, opts):
        """
        Parameters:
        device : torch.device
            Device for model training/testing
        save_dir : str
            Directory for reading/writing model checkpoints
        opts : dict
            Dictionary containing generator training options. Keys include:
                'x_dim' : int
                    Dimensionality of visual features
                'a_dim' : int
                    Dimensionality of semantic features (attributes)
                'z_dim' : int
                    Dimensionality of noise vector
                'hidden_dim' : int
                    Dimensionality of hidden layer, optional (default = 4096)
                lr' : float
                    Base learning rate 
                'lrdecay' : float
                    Learning rate decay factor, optional (default = 0)
                'wdecay' : float
                    Weight decay factor, optional (default = 0)
                'adam_beta1' : float
                    Beta1 parameter for Adam optimiser, optional (default = 0.9)
                'adam_beta2' : float
                    Beta2 parameter for Adam optimiser, optional (default = 0.999)
        """

        super(Generator, self).__init__(device, save_dir)

        self.device = device
        self._model_name = opts['name'] if 'name' in opts else 'generator'
        self._z_dim = opts['z_dim']

        #Architecture
        hidden_dim = opts['hidden_dim'] if 'hidden_dim' in opts else 4096
        self.fc1 = nn.Linear(opts['a_dim'] + self._z_dim,
                             hidden_dim,
                             bias=True)
        self.fc2 = nn.Linear(hidden_dim, opts['x_dim'], bias=True)

        self.to(self.device)

        self.apply(initialisers.init_weights_xavier)

        self.optimiser = self._get_optimiser(opts, self.parameters())
        self.lr_scheduler = self._get_lr_scheduler(opts, self.optimiser)

        #Load checkpoint (if one exists)
        self.load_checkpoint()

    def forward(self, a, z=None):
        """
        Generate visual feature from attribute and noise.

        Parameters:
        a : torch.Tensor
            Semantic features (attributes)
        z : torch.Tensor
            Normal Gaussian noise, optional

        Returns:
        x : torch.Tensor
            Visual features
        """

        #If noise isn't provided, sample from normal distribution
        if z is None:
            z = torch.randn(a.size(0), self._z_dim)

        x = F.leaky_relu(self.fc1(torch.cat((a, z), 1)), negative_slope=0.2)
        x = F.relu(self.fc2(x))

        return x

    def _generate_features(self, data_in, num_features):
        #Generate fake visual features. Call from public method generate_dataset().

        data_out = {
            'X': np.array([]),
            'Y': np.array([]),
            'A': {
                'continuous': np.array([])
            }
        }

        with torch.no_grad():
            for (input_a, input_y) in data_in:

                batch_a = torch.from_numpy(np.array(
                    [input_a] * num_features)).type(torch.FloatTensor)
                batch_y = torch.from_numpy(np.array(
                    [input_y] * num_features)).type(torch.FloatTensor)
                batch_z = torch.randn(num_features, self._z_dim)
                features = self.forward(batch_a.to(self.device),
                                        batch_z.to(self.device)).cpu().numpy()
                data_out['X'] = merge_array(data_out['X'], features)
                data_out['Y'] = merge_array(data_out['Y'], batch_y)
                data_out['A']['continuous'] = merge_array(
                    data_out['A']['continuous'], batch_a)

        return data_out

    def generate_dataset(self,
                         aug_file,
                         knn,
                         domain=['unseen'],
                         num_features=[200]):
        """
        Generate and save fake visual features.

        Parameters:
        aug_file : str
            File name for saving fake dataset
        knn : util.storage.container
            Loaded knn.h5 dataset (load using util.datasets.load())
        domain : list of str
            Domain of features to be generated (['unseen'], ['seen'] or['unseen','seen']), optional
        num_features: list of int
            Number of features to generate per class for each domain, optional
        """

        if (len(domain) > 1) and (len(num_features) == 1):
            num_features = [num_features[0]] * len(domain)

        new_dataset = {
            'train': {
                'X': np.array([]),
                'Y': np.array([]),
                'A': {
                    'continuous': np.array([])
                }
            },
            'info': {
                'num_features': str(num_features),
                'domain': str(domain)
            }
        }

        for _domain, _num in zip(domain, num_features):
            domain_in = {
                'unseen': zip(knn.zsl.data, knn.zsl.ids),
                'seen': zip(knn.openval.data, knn.openval.ids),
                'openset': zip(knn.openset.data, knn.openset.ids)
            }[_domain]

            print(":: Generating features [{}:{}]".format(_domain, _num))
            new_features = self._generate_features(domain_in, _num)
            new_dataset['train'] = merge_dict(new_dataset['train'],
                                              new_features)

        DataH5py().save_dict_to_hdf5(new_dataset, aug_file)

    def train(self, train_dataset, val_dataset=None, batch_size=64):
        raise NotImplementedError(
            "Train generator from GAN class - e.g. GAN.train()")

    def train_loop(self, train_loader):
        raise NotImplementedError("Train generator from GAN class")

    def validate(self, val_loader):
        raise NotImplementedError("Train generator from GAN class")

    def _get_training_details(self, opts):
        raise NotImplementedError("Call from GAN class")


class Discriminator(BaseModel):
    """
    A discriminator model class.
    Does not contain training methods - training is handled by the GAN class.
    ...

    Attributes:
    device : torch.device
        Device for model training/testing
    fc1 : torch.nn.Linear
        Fully connected layer 1
    fc2 : torch.nn.Linear
        Fully connected layer 2
    optimiser : torch.optim.Optimizer
        Optimiser to update discriminator weights
    lr_scheduler : object
        Learning rate scheduler for optimiser (from torch.nn.optim.lr_scheduler)

    Methods:
    forward(x, a)
        Get discriminator scores for visual features, conditioned on semantic attributes
    """

    def __init__(self, device, save_dir, opts):
        """
        Parameters:
        device : torch.device
            Device for model training/testing
        save_dir : str
            Directory for reading/writing model checkpoints
        opts : dict
            Dictionary containing discriminator training options. Keys include:
                'x_dim' : int
                    Dimensionality of visual features
                'a_dim' : int
                    Dimensionality of semantic features (attributes)
                'hidden_dim' : int
                    Dimensionality of hidden layer, optional (default = 4096)
                lr' : float
                    Base learning rate 
                'lrdecay' : float
                    Learning rate decay factor, optional (default = 0)
                'wdecay' : float
                    Weight decay factor, optional (default = 0)
                'adam_beta1' : float
                    Beta1 parameter for Adam optimiser, optional (default = 0.9)
                'adam_beta2' : float
                    Beta2 parameter for Adam optimiser, optional (default = 0.999)
        """

        super(Discriminator, self).__init__(device, save_dir)

        self.device = device
        self._model_name = opts['name'] if 'name' in opts else 'discriminator'

        #Architecture
        hidden_dim = opts['hidden_dim'] if 'hidden_dim' in opts else 4096
        self.fc1 = nn.Linear(opts['a_dim'] + opts['x_dim'],
                             hidden_dim,
                             bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

        self.to(self.device)

        self.apply(initialisers.init_weights_xavier)

        self.optimiser = self._get_optimiser(opts, self.parameters())
        self.lr_scheduler = self._get_lr_scheduler(opts, self.optimiser)

        #Load checkpoint (if one exists)
        self.load_checkpoint()

    def forward(self, x, a):
        """
        Get discriminator scores for visual features, conditioned on semantic attributes.

        Parameters:
        x : torch.Tensor
            Visual features
        a : torch.Tensor
            Semantic features (attributes)

        Returns:
        d : torch.Tensor
            Discriminator score
        """

        d = F.leaky_relu(self.fc1(torch.cat((x, a), 1)), negative_slope=0.2)
        d = self.fc2(d)

        return d

    def train(self, train_dataset, val_dataset=None, batch_size=64):
        raise NotImplementedError(
            "Train discriminator from GAN class - e.g. GAN.train()")

    def train_loop(self, train_loader):
        raise NotImplementedError("Train discriminator from GAN class")

    def validate(self, val_loader):
        raise NotImplementedError("Train discriminator from GAN class")

    def _get_training_details(self, opts):
        raise NotImplementedError("Call from GAN class")


class GAN(BaseModel):
    """
    A class for creating/training/testing/saving models.
    ...

    Attributes:
    classifier : Classifier
        Classifier model for performing validation during GAN training
    regressor : Regressor
        Regressor model for cycle loss in GAN
    generator : Generator
        Generator model
    discriminator : Discriminator
        Discriminator model

    Methods:
    forward(self, a, z=None)
        Generate visual feature from attribute and noise
    train(self, train_dataset, val_dataset=None)
        Train cycle consitent GAN (including validation classifier and regressor training)
    train_loop(train_loader)
        Perform one generator/discriminator training epoch
    validate(val_loader)
        Evaluate the generator by passing fake examples through validation classifier
    """

    def __init__(self, device, save_dir, opts):
        """
        Parameters:
        device : torch.device
            Device for model training/testing
        save_dir : str
            Directory for reading/writing model checkpoints
        opts : dict
            Dictionary containing GAN training options (including classifier, regressor,
            generator and discriminator option dictionaries). Keys include:
                'epochs' : int
                    Maximum number of GAN training epochs
                'batch_size' : int
                    Batch size for GAN training
                'cycle_lamda' : float
                    Scale term for cycle loss component
                '_gp_lambda' : float
                    Scale term for gradient penalty loss component
                'classifier' : dict
                    Dictionary containing validation classifier training options. See help(Classifier.__init__) for expected keys.
                'regressor' : dict
                    Dictionary containing regressor training options. See help(Regressor.__init__) for expected keys.
                'generator' : dict
                    Dictionary containing generator training options. See help(Generator.__init__) for expected keys.
                'discriminator' : dict
                    Dictionary containing discriminator training options. See help(Discriminator.__init__) for expected keys.
                'name' : str
                    Model name for saving/print, optional (default = 'GAN')
                'val_freq' : int
                    Frequency of GAN validation during training (epochs), optional (default = 5)
                'save_freq' : int
                    Frequency of GAN checkpointing during training (epochs), optional (default = 5)
        """

        super(GAN, self).__init__(device, save_dir)

        #Classifier is not used for GAN training or GZSL evaluation - It is used as a stopping condition for GAN training
        self.classifier = Classifier(device, save_dir, opts['classifier'])
        self.regressor = Regressor(device, save_dir, opts['regressor'])
        self.generator = Generator(device, save_dir, opts['generator'])
        self.discriminator = Discriminator(device, save_dir,
                                           opts['discriminator'])

        self._z_dim = opts['generator']['z_dim']
        self._model_name = opts['name'] if 'name' in opts else 'GAN'

        #Set-up for training
        self._get_training_details(opts)
        self._cyc_lambda = opts['cycle_lambda']
        self._gp_lambda = opts['gp_lambda']
        self._batch_size_cls = opts['classifier']['batch_size']
        self._batch_size_reg = opts['regressor']['batch_size']
        self._batch_size_GAN = opts['batch_size']
        self._start_epoch = self.generator._start_epoch

    def forward(self, a, z=None):
        """
        Generate visual feature from attribute and noise.

        Parameters:
        a : torch.Tensor
            Semantic features (attributes)
        z : torch.Tensor
            Normal Gaussian noise, optional

        Returns:
        x : torch.Tensor
            Visual features
        """

        self.generator(a, z)

    def train(self, train_dataset, val_dataset=None):
        """
        Train cycle consitent GAN (including validation classifier and regressor training).

        Parameters:
        train_dataset : util.loaders.LoaderH5
            Dataset class for training data.
        val_dataset : util.loaders.LoaderH5
            Dataset class for validation data, optional

        Returns:
        x : torch.Tensor
            Visual features
        """

        #Train validation classifier for monitoring GAN training
        self.classifier.train(train_dataset,
                              val_dataset,
                              batch_size=self._batch_size_cls)

        #Train regressor for cycle loss in GAN training
        self.regressor.train(train_dataset,
                             val_dataset,
                             batch_size=self._batch_size_reg)

        #Train generator/discriminator
        super(GAN, self).train(train_dataset,
                               val_dataset,
                               batch_size=self._batch_size_GAN)

    def _step_discriminator(self, a, x):
        #Training step for discriminator - called from self.train_loop()

        fake_d = self.generator(
            a,
            torch.randn(a.size(0), self._z_dim).to(self.device))
        fake_scores_d = torch.mean(self.discriminator(fake_d.detach(), a))
        real_scores_d = torch.mean(self.discriminator(x, a))

        gradient_penalty = self._get_gradient_penalty(x, fake_d, a,
                                                      self._gp_lambda)

        self.discriminator.zero_grad()
        d_loss = fake_scores_d - real_scores_d + gradient_penalty
        d_loss.backward()
        self.discriminator.optimiser.step()

        self.discriminator.lr_scheduler.step()

        return d_loss.item()

    def _step_generator(self, a):
        #Training step for generator - called from self.train_loop()

        fake_g = self.generator(
            a,
            torch.randn(a.size(0), self._z_dim).to(self.device))
        fake_scores_g = self.discriminator(fake_g, a)
        cyc_loss = F.mse_loss(self.regressor(fake_g), a)
        g_loss = -torch.mean(fake_scores_g) + cyc_loss * self._cyc_lambda

        self.generator.zero_grad()
        g_loss.backward()
        self.generator.optimiser.step()

        self.generator.lr_scheduler.step()

        return g_loss.item()

    def train_loop(self, train_loader):
        """
        Perform one generator/discriminator training epoch - overrides BaseModel.train_loop()

        Parameters:
        train_loader : torch.utils.data.Dataloader
            Dataloader for training data
        Returns:
         : list
            Training loss as ['G Loss', float, 'D Loss', float]
        """

        g_loss_running = 0.0
        d_loss_running = 0.0
        cyc_loss_running = 0.0

        for i, data in enumerate(train_loader):

            x = data['x'].to(self.device)
            a = data['a'].to(self.device)

            d_loss_running += self._step_discriminator(a, x)

            g_loss_running += self._step_generator(a)

        return [
            "G Loss", g_loss_running / float(len(train_loader)), "D Loss",
            d_loss_running / float(len(train_loader))
        ]

    def _get_gradient_penalty(self, real_data, generated_data, a, _gp_lambda):
        #Gradient penalty term for discriminator loss - called from self._step_discriminator()

        batch_size = real_data.size()[0]

        alpha = torch.rand(batch_size, 1).expand_as(real_data).to(self.device)
        interpolated = autograd.Variable(alpha * real_data +
                                         (1 - alpha) * generated_data,
                                         requires_grad=True).to(self.device)

        interpolated_d = self.discriminator(interpolated, a)

        gradients = autograd.grad(outputs=interpolated_d,
                                  inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      interpolated_d.size()).to(self.device),
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

        return _gp_lambda * ((gradients_norm - 1)**2).mean()

    def validate(self, val_loader):
        """
        Evaluate the generator by passing fake examples through validation classifier.
        Overrides BaseModel.validate().

        Parameters:
        val_loader : torch.utils.data.Dataloader
            Dataloader for validation data
        Returns:
         : list
            Validation accuracy as ['Accuracy', float]
        """

        #Initialised in accuracy function
        correct = None
        total = None

        with torch.no_grad():
            for i, data in enumerate(val_loader):

                a = data['a'].to(self.device)
                x_fake = self.generator(
                    a,
                    torch.randn(a.size(0), self._z_dim).to(self.device))

                scores = self.classifier(x_fake)
                correct, total = self.classifier._class_accuracy_accum(
                    scores, data['y'].to(self.device), correct, total)

        return [
            "Accuracy (fake, seen)",
            (correct[total > 0] /
             (total[total > 0]).type(torch.FloatTensor)).mean()
        ]

    def _save_checkpoint(self, epoch, losses):
        #Save training checkpoint for  both generator and discriminator - overrides BaseModel._save_checkpoint().
        self.generator._save_checkpoint(epoch, losses[0])
        self.discriminator._save_checkpoint(epoch, losses[1])

    def load_checkpoint(self):
        raise NotImplementedError(
            "Load sub-models from within themselves - e.g. self.generator.load_epoch()"
        )
