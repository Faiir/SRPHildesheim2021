
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

# TODO

- Test more datasets !low (Whoever feels like doing it)
- Include spectral norm -res / jacobian penalty -conv - medium (Abdur Niklas)
- Merge Branches !High (Niklas)
- include DDU Experiment Setup !medium (Era & Sam)
- Include density plots !medium (Markus)
