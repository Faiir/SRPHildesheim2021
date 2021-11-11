
# SRP-Hildesheim2021

# Set up enviornment

``` #!/bin/bash
pip install -r requirements.txt
```

# Run Experiment

In colab

```#!/bin/bash

%cd ./project
python -m setup_experiment -c "path\to\config\.json"
```

# Experiment settings

We construct the manager for the active learning experiment. It will use the same starting data for every experiment in the chain

```json


  "experiments": [
    {
        // settings will be shared accroos the experiment
    "basic_settings": {
    "oracle_stepsize": 50, // no. of samples per AL iteration
    "oracle_steps": 20, // how many AL iterations
    "iD": "CIFAR10", // in-distribution dataset
    "OoD": ["FashionMNIST", "MNIST"], // SVHN CIFAR100, Soon: GrayScale Cifar, Subclass Cifar
    "labelled_size": 100, // starting training pool
    "pool_size": 5000, // unlabeled pool data
    "OOD_ratio": 0.15, // no. of OoD data in the unlabelled pool
    // training settings
    "epochs": 10,
    "batch_size": 64,
    "weight_decay": 1e-4,
    "metric": "accuracy",
    "lr": 0.1,
    "nesterov": false,
    "momentum": 0.9,
    "lr_sheduler": true,
    "num_classes": 10,
    "validation_split": 0.1,
    "validation_source": "test",
    "criterion": "crossentropy",
    "verbose": 1
    },
    // specific settings to the different methods 
    "exp_settings": [
        {
          "exp_type": "baseline",
          "exp_name": "funny_name_here",
          "plots": false,
          "model": "base",
          "oracle": "highest-entropy"
        },
        {
          "exp_type": "baseline-ood",
          "exp_name": "funny_name_here2",
          "oracle": "highest-entropy",
          "model": "base",
          "plots": false
        },
        {
          "exp_type": "extra_class",
          "exp_name": "funny_name_here",
          "extra_class-type": "hard / soft",
          "extra_class_thresholding": 0.1,
          "oracle": "highest-entropy",
          "model": "base",
          "plots": false
        },
        {
          "exp_type": "gram",
          "exp_name": "funny_name_here",
          "oracle": "highest-entropy",
          "model": "gram_resnet",
          "plots": false
        },
        {
          "exp_type": "looc",
          "exp_name": "funny_name_here",
          "similarity": "E",
          "scaling_factor": "G / R",
          "oracle": "highest-entropy",
          "bugged_and_working": true,
          "do_pertubed_images": true,
          "model": "LOOC",
          "plots": false,
          "plotsettings": {
            "do_plot": true,
            "density_plot": true,
            "layer_plot": true
          }
        },
        {
          "exp_type": "genodin",
          "exp_name": "funny_name_here",
          "similarity": "C",
          "oracle": "highest-entropy",
          "do_pertubed_images": true,
          "scaling_factor": "G / R",
          "model": "GenOdin",
          "bugged_and_working": false,
          "plotsettings": {
            "do_plot": true,
            "density_plot": true,
            "layer_plot": true
          }
        }
        {
          "exp_type": "ddu",
          "exp_name": "funny_name_here",
          "plots": false,
          "model": "DDU",
          "oracle": "ddu-sampler",
          "spectral_normalization": true,
          "temp": 1.0
        }
      ]
    }
  ]



```

# TODO

- layer analysis
- other datasets
- Gram-Class
- experiments
- add larger model option for cifar100
- train function to train the final networkn (add model, trainfunction, test function  / scores for this to report -> should load the datasets based on the statusmanager)