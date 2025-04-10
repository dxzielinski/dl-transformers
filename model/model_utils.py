"""
Utility functions and classes for model training.

Main focus: implementing dynamic transformer model architecture.

Default, fixed hyperparameters are:

- optimizer: AdamW
- loss function: CrossEntropyLoss
- learning rate schedule: Noam or linear warmup with cosine decay or ReduceLROnPlateau
- learning rate - it will be found by learning rate tuning
- activation function: RELU

Hyperparameter space for the model includes:

- number of layers [4 to 10]
- number of attention heads [4 to 12]
- hidden size [256 to 768]
- dropout rate [0.05 to 0.15]
- weight decay [0 to 0.01]
"""
