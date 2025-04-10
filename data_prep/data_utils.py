"""
Utility functions and classes for data preprocessing.

Flow:

1. Convert audio files to spectrograms in separate data pipeline.
Make sure it is grayscale with proper size (Resize with preserving
aspect ratio and bilinear interpolation if needed). Remove silence!
Then, save the spectrograms as a dataset.

2. Transforms for spectrograms obtained from audio include:

- time masking
- frequency masking
- noise injection
- normalization with mean and std from the train set

Note: time masking, frequency masking and noise injection should be
wrapped in RandomApply to always be applied with a certain probability.
"""
