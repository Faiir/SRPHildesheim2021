
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

- test_size: test_set
- pool_size: unlabeled pool size
- labelled_size: initial dataset size
- OOD_ratio: % of OOD in unlabeled Pool
- oracle_stepsize: Num. of added images to trainset (sampled OOD gets discarded)
- oracle_steps: num: active learning cycles
- epochs
- batch_size
- oracles (list): types of sampling meachnisms (currently: random, least confidence, highest entropy)
- weight_deacy
- metric: main eval_metric (f1,acc,auroc)
- model_name: model type (base, gen_odin_conv/resnet, DDU )
- similarity: genodin measure (C,E,I)
- include_bn: batchnorm
- datasets

# TODO

- Test more datasets !low (Whoever feels like doing it)
- Include spectral norm -res / jacobian penalty -conv - !medium (Abdur Niklas)
- Merge Branches !High (Niklas) ---
- include DDU Experiment Setup !medium (Era & Sam)
- Include density plots !medium (Markus)
- sampler class !high (Abdur)
- stratified sampling datamager !high
- dataset selection modular !medium (Niklas)
- 11 Class integration in different exp setup !low (Niklas)
