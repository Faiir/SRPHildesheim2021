
# SRP-Hildesheim2021 - Robust Active Learning

# Set up enviornment

pytorch <= 1.10

``` #!/bin/bash
pip install -r requirements.txt
```

# Run Experiment

In colab

```#!/bin/bash

python -m setup_experiment -c "path\to\config\.json"

```

# Experiment settings

We construct the manager for the active learning experiment. It will use the same starting data for every experiment in the chain

```json
{
  "experiments": [
    {
      "basic_settings": {
        "oracle_stepsize": 50,
        "oracle_steps": 10,
        "iD": "CIFAR10", // CIFAR100
        "OoD": ["FashionMNIST", "MNIST"], //CIFAR10_ood, CIFAR100_ood , SVHN
        "grayscale": false,
        "subclass": {
          "do_subclass": false,
          "iD_classes": [
            0, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23,
            24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44,
            46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 66,
            67, 68, 69, 71, 72, 73, 74, 76, 77, 78, 79, 81, 82, 83, 84, 86, 87,
            88, 89, 91, 92, 93, 94, 96, 97, 98, 99
          ],
          "OoD_classes": [
            1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
            85, 90, 95
          ]
        },
        "labelled_size": 1000,
        "pool_size": 30000,
        "OOD_ratio": 0.5,
        "epochs": 200,
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
        "verbose": 2
      },
      "exp_settings": [
        {
          "exp_type": "genodin",
          "exp_name": "genodin-cifar-015",
          "similarity": "C",
          "oracle": "highest-entropy",
          "do_pertubed_images": true,
          "scaling_factor": "R",
          "model": "GenOdin",
          "bugged_and_working": false,
          "plotsettings": {
            "do_plot": false,
            "density_plot": false,
            "layer_plot": false
          },
          "perform_experiment": false
        },
        {
          "exp_type": "looc",
          "exp_name": "looc-cifar-015",
          "similarity": "C",
          "oracle": "LOOC",
          "do_pertubed_images": true,
          "scaling_factor": "R",
          "model": "LOOC",
          "bugged_and_working": false,
          "plotsettings": {
            "do_plot": false,
            "density_plot": false,
            "layer_plot": false
          },
          "perform_experiment": false
        },
        {
          "exp_type": "ddu",
          "exp_name": "ddu-cifar-015",
          "plots": false,
          "model": "DDU",
          "oracle": "ddu-sampler",
          "spectral_normalization": true,
          "temp": 1.0,
          "perform_experiment": false
        },
        {
          "exp_type": "baseline",
          "exp_name": "baseline",
          "oracle": "baseline",
          "model": "base",
          "plots": false,
          "perform_experiment": false
        },
        {
          "exp_type": "baseline-ood",
          "exp_name": "baseline-ood",
          "oracle": "baseline",
          "model": "base",
          "plots": false,
          "perform_experiment": false
        },
        {
          "exp_type": "extra_class",
          "exp_name": "extra_class_hard",
          "extra_class_thresholding": "hard",
          "oracle": "highest-entropy",
          "model" : "base",
          "perform_experiment": false
        },
        {
          "exp_type": "extra_class",
          "exp_name": "extra_class_soft",
          "extra_class_thresholding": "soft",
          "oracle": "highest-entropy",
          "model" : "base",
          "perform_experiment": false
        },
        {
          "exp_type": "gram",
          "exp_name": "gram-exp",
          "oracle": "highest-entropy",
          "model": "gram_resnet",
          "plots": false,
          "perform_experiment": true
        }
      ]
    }
  ]
}

```

# TODO

- layer analysis
- add larger model option for cifar100
- write down experiment setup and send to supervisors ()

# Dataset Options

- cifar10 - mnist, fmnist, (svhn)
- grayscaled cifar - mnist and fmnist (grayscaled svhn)
- cifar10 - sublass 0-6 iD and 7-9 Ood
- cifar100 - subclassed -> per superclass (20) vs other 80 iD classes

# AL+OoD Experiments

min. 3x per experiment version - per AL-OoD

- 15% OoD
- 30% OoD

- 50%  grayscaled cifar - mnist and fmnist (grayscaled svhn)

# Presentation

30 mins of presentation

- Motivation  (Done)
- Research Question (Done )
- Related Work / Explanation of Methods -> Expand on DDU and Looc (theory) (50%)
- Looc-Layer Analysis (Abdur)
- Experiments (Setup, Metrics, Good and Bad results) (All of us)
- Conclusion (whole thing)
- Link to Profiles / Github whatever
