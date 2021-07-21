import torch
import models
import arguments
import os
from util import datasets
from util.storage import Container, DataH5py
import util.loaders as loaders
import numpy as np
from sklearn.model_selection import train_test_split
import json
import datetime
from pathlib import Path

def get_data_as_dict(x = None, y = None, a=None):
	return {'x': x, 'y': y, 'a': a}

def get_split_datasets(dataset, val_size=0.1):

    #Split data into training/validation (1-val_size/val_size)
    split = train_test_split(dataset.X, dataset.Y, dataset.A.continuous, test_size=val_size, random_state=42)

    #Create dataset objects for both training and validation splits
    train_data = get_data_as_dict(split[0], split[2], split[4])
    train_dataset = loaders.LoaderH5(train_data)
    val_data = get_data_as_dict(split[1], split[3], split[5])
    val_dataset = loaders.LoaderH5(val_data)

    return train_dataset, val_dataset

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def unique_work_dir():
    #Return unique working directory from date and time
    d = datetime.datetime.now()
    return "exp_{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}".format(d.year,d.month,d.day,d.hour,d.minute,d.second)

def setup_model(model, device, model_dir, opts):
    create_dir(model_dir)
    return model(device, model_dir, opts)

def train_GAN(device, gan_dir, opts, dataset):

    #Set-up directory and create GAN
    gan = setup_model(models.GAN, device, gan_dir, opts)
    
    #Get train/val datasets
    train_dataset, val_dataset = get_split_datasets(dataset)

    #Train GAN
    gan.train(train_dataset, val_dataset)

    return gan

def generate_fake_data(gan, knn, aug_file, domain, num_features):

    #Set-up directory
    create_dir(str(Path(aug_file).parent))

    #Used trained generator to synthesis fake visual samples - save to aug_file
    gan.generator.generate_dataset(aug_file, knn, domain, num_features)

def train_GZSL_classifier(device, classifier_dir, opts, dataset):

    #Set-up directory and create GZSL classifier
    gzsl_classifier = setup_model(models.Classifier, device, classifier_dir, opts)

    #Get train/val datasets
    train_dataset, val_dataset = get_split_datasets(dataset)

    #Train a classifier for GZSL on the provided dataset
    gzsl_classifier.train(train_dataset, val_dataset, batch_size=opts['batch_size'])

    return gzsl_classifier

def harmonic_mean(a, b):
    return 2*a*b/(a+b)

def test_GZSL_classifier(gzsl_classifier, opts, dataset, knn):

    #Get seen/unseen test data
    test_dataset_seen = get_split_datasets(dataset.seen, val_size=0)[0]
    test_dataset_unseen = get_split_datasets(dataset.unseen, val_size=0)[0]

    #Per-class accuracies
    batch_size = opts['batch_size'] if 'batch_size' in opts else 64
    seen_acc = gzsl_classifier.class_accuracy(test_dataset_seen, batch_size, knn.openval.ids-1)
    unseen_acc = gzsl_classifier.class_accuracy(test_dataset_unseen, batch_size, knn.zsl.ids-1)

    print("Y(U) = {0:.4f}, Y(S) = {1:.4f}, H = {2:.4f}, ".format(unseen_acc, seen_acc, harmonic_mean(seen_acc, unseen_acc)))

    return unseen_acc, seen_acc

if __name__ == '__main__':

    #Parse command line options
    args = arguments.Arguments().parse()

    #Extract configuration file
    with open(args.config) as f:
        opts = json.load(f)

    #Set GPU ID or cpu only mode
    if args.cpu:
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device('cuda:0')

    #Load dataset
    dataset, knn = datasets.load(opts['data_dir'])

    #Set-up experiment dir
    if args.work_dir is None:
        work_dir = "experiments/" + opts['dataset'] + "/" + unique_work_dir()
    else:
        work_dir = args.work_dir

    #Train multi-modal cycle-consistent GAN
    #Includes a validation classifier to monitor GAN training, a regressor and a generator/discriminator
    if args.train_GAN:
        gan = train_GAN(device, work_dir + "/GAN", opts['GAN'], dataset.train)

    #Data augmentation file
    aug_file = work_dir + "/generated_data/data.h5"

    #Generate a dataset of fake visual samples
    if args.generate:

        #Load trained GAN model if not just trained
        if not args.train_GAN:
            gan = setup_model(models.GAN, device, work_dir + "/GAN", opts['GAN'])

        #Create fake data
        generate_fake_data(gan, knn, aug_file, args.domain, args.num_features)

    #Augment dataset and train GZSL classifier
    if args.train_GZSL:

        #Augment real dataset with fake data
        if args.aug_op is not 'none':
            dataset = datasets.augment_dataset(dataset, aug_file, args.aug_op) 

        #Train GZSL classifier
        classifier = train_GZSL_classifier(device, work_dir + "/GZSL_classifier", opts['GZSL_classifier'], dataset.train)

    #Test GZSL classifier
    if args.train_GZSL or args.test_GZSL:

        #Load trained classifier if not just trained
        if not args.train_GZSL:
            classifier = setup_model(models.Classifier, device, work_dir + "/GZSL_classifier", opts['GZSL_classifier'])

        print(":: Evaluating GZSL classifier on test data")
        accuracies = test_GZSL_classifier(classifier, opts['GZSL_classifier'], dataset.test, knn)

    if not any([args.train_GAN, args.generate, args.train_GZSL, args.test_GZSL]):
        print("Nothing to do! Set command line arguments to train/generate/test!")
