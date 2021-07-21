<p align=center><strong>~Please note this is only a <em>beta</em> release at this stage~</strong></p>

# Cycle-WGAN: Multi-modal Cycle-consistent Generalised Zero-Shot Learning

[![Best of ACRV Repository](https://img.shields.io/badge/collection-best--of--acrv-%23a31b2a)](https://roboticvision.org/best-of-acrv)
![Primary language](https://img.shields.io/github/languages/top/best-of-acrv/cycle-wgan)
[![PyPI package](https://img.shields.io/pypi/pyversions/cycle-wgan)](https://pypi.org/project/cycle-wgan/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/cycle_wgan.svg)](https://anaconda.org/conda-forge/cycle_wgan)
[![Conda Recipe](https://img.shields.io/badge/recipe-cycle_wgan-green.svg)](https://anaconda.org/conda-forge/cycle_wgan)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/cycle_wgan.svg)](https://anaconda.org/conda-forge/cycle_wgan)
[![License](https://img.shields.io/github/license/best-of-acrv/cycle-wgan)](./LICENSE.txt)

Cycle-WGAN is a generalised zero-shot learning (GZSL) semantic classifier that improves performance through multi-modal cycle-consistent feature generation. Synthesised representations of the seen and unseen classes are used to train a GZSL classifier, with the consistency of cycling between these representations and semantic features guaranteed.

TODO: image of the system's output

This repository contains an open source implementation of Cycle-WGAN in Python, with training configuration for four common datasets: CUB, FLO, SUN, and AWA1. The package provides PyTorch implementations for training, evaluation, and prediction in your own systems. The package is easily installable with `conda`, and can also be installed via `pip` if you prefer manually managing system dependencies.

Our code is free to use, and licensed under BSD-3. We simply ask that you [cite our work](#citing-our-work) if you use Cycle-WGAN in your own research.

## Related resources

This repository brings the work from a number of sources together. Please see the links below for further details:

- our original paper: ["Multi-modal Cycle-consistent Generalized Zero-Shot Learning"](#citing-our-work)
- original TensorFlow implementation: [https://github.com/rfelixmg/frwgan-eccv18](https://github.com/rfelixmg/frwgan-eccv1o8)
- utility code incorporated into this repository: [https://github.com/rfelixmg/util](https://github.com/rfelixmg/util)

## Installing Cycle-WGAN

We offer three methods for installing Cycle-WGAN:

1. [Through our Conda package](#conda): single command installs everything including system dependencies (recommended)
2. [Through our pip package](#pip): single command installs Cycle-WGAN and Python dependences, you take care of system dependencies
3. [Directly from source](#from-source): allows easy editing and extension of our code, but you take care of building and all dependencies

### Conda

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and [NVIDIA drivers installed](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu&target_version=20.04&target_type=deb_network) if you want CUDA acceleration. We provide Conda packages through [Conda Forge](https://conda-forge.org/), which recommends adding their channel globally with strict priority:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once you have access to the `conda-forge` channel, Cycle-WGAN is installed by running the following from inside a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
u@pc:~$ conda install cycle-wgan
```

TODO confirm what role CUDA plays in Cycle-WGAN

We don't explicitly lock the PyTorch installation to a CUDA-enabled version to maximise compatibility with our users' possible setups. If you wish to ensure a CUDA-enabled PyTorch is installed, please use the following installation line instead:

```
u@pc:~$ conda install pytorch=*=*cuda* cycle-wgan
```

You can see a list of our Conda dependencies in the [Cycle-WGAN feedstock's recipe](https://github.com/conda-forge/cycle_wgan-feedstock/blob/master/recipe/meta.yaml).

### Pip

Before installing via `pip`, you must have the following system dependencies installed if you want CUDA acceleration:

- NVIDIA drivers
- CUDA

Then Cycle-WGAN, and all its Python dependencies can be installed via:

```
u@pc:~$ pip install cycle-wgan
```

TODO something about the building of custom layers with CUDA...

### From source

Installing from source is very similar to the `pip` method above

TODO validate this statement is actually true "due to Cycle-WGAN only containing Python code".

Simply clone the repository, enter the directory, and install via `pip`:

```
u@pc:~$ pip install -e .
```

TODO check this actually handles building of the custom layers with CUDA

_Note: the editable mode flag (`-e`) is optional, but allows you to immediately use any changes you make to the code in your local Python ecosystem._

We also include scripts in the `./scripts` directory to support running FCOS without any `pip` installation, but this workflow means you need to handle all system and Python dependencies manually.

## Using Cycle-WGAN

### Cycle-WGAN from the command line

### Cycle-WGAN Python API

## Citing our work

If using Cycle-WGAN in your work, please cite [our original ECCV paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/RAFAEL_FELIX_Multi-modal_Cycle-consistent_Generalized_ECCV_2018_paper.pdf):

```bibtex
@inproceedings{felix2018multi,
  title={Multi-modal cycle-consistent generalized zero-shot learning},
  author={Felix, Rafael and Reid, Ian and Carneiro, Gustavo and others},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={21--37},
  year={2018}
}
```

TODO delete everything from here down when finished

This repository contains a PyTorch implementation of the ECCV'18 paper "Multi-modal Cycle-consistent Generalized Zero-Shot Learning" - [link to paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/RAFAEL_FELIX_Multi-modal_Cycle-consistent_Generalized_ECCV_2018_paper.pdf).

The original TensorFlow implementation can be found by clicking this [link](https://github.com/rfelixmg/frwgan-eccv18). This repository also contains some utility code originally published [here](https://github.com/rfelixmg/util).

## Citation

```
@inproceedings{felix2018multi,
  title={Multi-modal Cycle-Consistent Generalized Zero-Shot Learning},
  author={Felix, Rafael and Kumar, BG Vijay and Reid, Ian and Carneiro, Gustavo},
  booktitle={European Conference on Computer Vision},
  pages={21--37},
  year={2018},
  organization={Springer}
}
```

## Set-up

The following set-up assumes you are using an Ubuntu system.

### Anaconda Set-up

We use a conda virtual environment for this implementation.
If you do not have `conda` or [Anaconda](https://www.anaconda.com/distribution/#linux) installed, enter the following commands into your terminal:

```bash
# The version of Anaconda may be different depending on when you are installing`

$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ sh Miniconda3-latest-Linux-x86_64.sh

# and follow the prompts. Make sure to select yes for running conda init, otherwise the defaults are generally good.`

```

_You may have to open a new terminal or re-source your ~/.bashrc to get access to the conda command._

If don't want conda's base environment to be activated on startup, set the auto_activate_base parameter to false:

```bash
$ conda config --set auto_activate_base false
```

### Clone repository

Enter the following commands to clone this repository:

```bash
$ git clone https://github.com/Best-of-ACRV/cycle_consistent_GZSL.git
$ cd cycle_consistent_GZSL/
```

### Virtual Environment

The conda virtual environment is created from the .yml file _virtual_environment/pytorch_gzsl.yml_ - enter the following command to create the environment:

```bash
$ conda env create -f virtual_environment/pytorch_gzsl.yml
```

This will create a conda environment named _pytorch_gzsl_. Check that the environment was created successfully:

```bash
$ conda info --envs
```

## Datasets

The four datasets analysed in the paper (CUB, FLO, SUN and AWA1) are available in h5 format. We provide a script to download and unzip to the expected directory (_data/_). The download is ~1.3GB and the unzipped data will take up about ~3.1GB. To download and unpack the data, run the following from the cycle_consistent_GZSL root directory:

```bash
$ bash data/download_datasets.sh
```

Alternatively, the datasets can be downloaded via this [link](https://drive.google.com/file/d/1cJ-Hl5F9LOn4l-53vhu3-zfv0in2ahYI/view).

## Usage

Ensure your working directory is set to repository base directory (cycle_consistent_GZSL) and activate the virtual environment:

```bash
$ conda activate pytorch_gzsl
```

You should now see (pytorch_gzsl) in your terminal prompt:

```bash
(pytorch_gzsl) $
```

### Basic Usage

Training the cycle-consistent GZSL method consists of the following steps:

1. GAN training.
   1. Pre-train validation classifier for monitoring generator training.
   2. Pre-train regressor (used for cycle loss component).
   3. Train generator/discriminator.
2. Generating fake visual features from unseen (and optionally, seen) classes.
3. Training a GZSL classifier on the fake visual features (or a combination of fake/real).

Model classes (including training/validation/testing routines) are found in models.py - classes include Classifier (for the validation classifier and GZSL classifier), Regressor, Generator, Discriminator and GAN (which has an instantiation of the other four classes as attributes). The model classes expect a dictionary of training/model options. We provide configuration .json files for each of the four datasets in the _configs/_ directory. Full details on the model/training configuration options can be found by entering the following commands:

```bash
(pytorch_gzsl) $ python
>>> import models
>>> help(models.GAN.__init__)
>>> help(models.Generator.__init__)
>>> help(models.Discriminator.__init__)
>>> help(models.Regressor.__init__)
>>> help(models.Classifier.__init__)
```

In general, experiments are run by calling the run.py file. To view the expected arguments, enter:

```bash
(pytorch_gzsl) $ python run.py --help
```

The example command below runs all training steps outlined above (with default settings), where CONFIG_JSON is a model/training configuration .json file and GPU_ID is the device ID to be used.

```bash
(pytorch_gzsl) $ python run.py --config CONFIG_JSON --gpu GPU_ID --train-gan --gen-fake --train-cls
```

### Pre-defined Experiments

We provide bash scripts to run experiments on each of the CUB, FLO, SUN and AWA datasets with the configurations/settings used to produce the results in the paper. For example, the _scripts/cub_ directory contains the following files:

1. **run_all.sh**: Runs a complete GZSL experiment, including training the GAN, generating a fake dataset, training a GZSL classifier on the real and/or fake data and evaluating the GZSL classifier on the test data.
2. **run_train_gan.sh**: Run GAN training only.
3. **run_gen_fake.sh**: Generates a fake dataset using a trained generator - change the WORKDIR to the correct directory.
4. **run_train_cls.sh**: Trains a GZSL on real and/or fake data - change the WORKDIR to the correct directory.
5. **run_test_cls.sh**: Evaluate a trained GZSL classifier on the test data - change the WORKDIR to the correct directory.

Each script takes an optional argument to specify the GPU ID - if not ID is provided, GPU 0 is used.

To run the complete pre-defined experiment on the CUB dataset enter the following command:

```bash
(pytorch_gzsl) $ bash scripts/cub/run_all.sh GPU_ID #Replace GPU_ID with desired device ID
```
