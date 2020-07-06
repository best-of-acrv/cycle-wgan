# Multi-modal Cycle-consistent Generalized Zero-Shot Learning
This repository contains a PyTorch implementation of the ECCV'18 paper "Multi-modal Cycle-consistent Generalized Zero-Shot Learning" -  [link to paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/RAFAEL_FELIX_Multi-modal_Cycle-consistent_Generalized_ECCV_2018_paper.pdf).

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

*You may have to open a new terminal or re-source your ~/.bashrc to get access to the conda command.*

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
The conda virtual environment is created from the .yml file *virtual_environment/pytorch_gzsl.yml* - enter the following command to create the environment:
```bash
$ conda env create -f virtual_environment/pytorch_gzsl.yml
```
This will create a conda environment named *pytorch_gzsl*. Check that the environment was created successfully:
```bash
$ conda info --envs
```

## Datasets
The four datasets analysed in the paper (CUB, FLO, SUN and AWA1) are available in h5 format. We provide a script to download and unzip to the expected directory (*data/*). The download is ~1.3GB and the unzipped data will take up about ~3.1GB. To download and unpack the data, run the following from the cycle_consistent_GZSL root directory:
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

Model classes (including training/validation/testing routines) are found in models.py - classes include Classifier (for the validation classifier and GZSL classifier), Regressor, Generator, Discriminator and GAN (which has an instantiation of the other four classes as attributes). The model classes expect a dictionary of training/model options. We provide configuration .json files for each of the four datasets in the *configs/* directory. Full details on the model/training configuration options can be found by entering the following commands:
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
We provide bash scripts to run experiments on each of the CUB, FLO, SUN and AWA datasets with the configurations/settings used to produce the results in the paper. For example, the *scripts/cub* directory contains the following files:
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
