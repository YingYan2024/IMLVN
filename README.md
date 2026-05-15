# Interpretable Multi-Layer Voting Network for Fault Diagnosis

This folder contains the MATLAB implementation used to train and evaluate an
Interpretable Multi-Layer Voting Network (IMLVN/MLVT) for HVAC air handling
unit fault diagnosis. The main training script is `MLVT.m`.

The method follows the paper *A Novel Interpretable Multi-Layer Voting Network
for Fault Diagnosis*. Instead of treating diagnosis as a black-box
classification problem, the network converts sensor residuals into signed voting
evidence and then aggregates these votes to estimate the most likely fault
class. The model is designed so that its intermediate layers have physical
meaning: polynomial feature approximation, residual-based fault features, voting
value generation, feature-to-fault voting, voting analysis, and final
probability output.

## Experiment

The experiment targets fault diagnosis for an HVAC air handling unit (AHU). The
dataset used in the paper is based on the ASHRAE RP-1312 AHU data and contains
14 classes: 13 fault categories and one normal category. The code assumes that
the samples are arranged class by class and uses ten-fold cross-validation for
evaluation.

In `MLVT.m`, the default workflow is:

1. Load the input feature matrix from `Data2.mat`.
2. Create class labels for 14 classes.
3. Split the data with 10-fold cross-validation.
4. Normalize each fold using statistics from the training split.
5. Expand the inputs with polynomial terms.
6. Train the multi-layer voting network by gradient descent.
7. Report accuracy, class-wise accuracy, precision, recall, F1 score, and
   average runtime.

## Main Files

- `MLVT.m`: main training and evaluation script.
- `Data.mat`, `Data2.mat`: MATLAB data files containing the feature matrix used
  by the training script.
- `Taylor_expan.m`: polynomial feature expansion.
- `Initialization.m`: weight initialization methods.
- `Activate.m`, `Activate_grad.m`: activation functions and derivatives.
- `Edifference.m`, `Edifference_grad.m`: residual/fault-feature calculation and
  derivatives.
- `Gaussian_trans.m`, `Gaussian_trans_grad.m`: optional residual transforms or
  normalization layers and derivatives.
- `Gradient_renewal.m`: optimizer update rules, including AdaGrad, RMSprop,
  momentum, and Adam-style updates.
- `forward_propagation.m`: forward pass used during training and testing.

Plotting utilities and generated figures are not described here because they are
not required for running the main training experiment.

## Requirements

- MATLAB, tested with the local MATLAB installation used for this project.
- Statistics and Machine Learning Toolbox, for functions such as `cvpartition`
  and `confusionmat`.
- Parallel Computing Toolbox or a compatible GPU setup, because `MLVT.m` uses
  `gpuArray`.
- Deep Learning Toolbox or another available implementation of `softmax`.

If GPU execution is not available, remove or replace the `gpuArray(...)` calls
in `MLVT.m` and related data assignments.

## How to Run

Open MATLAB, change the working directory to this folder, and run:

```matlab
MLVT
```

The default important settings near the top of `MLVT.m` are:

- `testNum = 1`: number of repeated full experiments.
- `numEpochs = 1000`: training epochs for each fold.
- `lr = 0.005`: learning rate.
- `k = 10`: number of cross-validation folds.
- `Power = 2`: polynomial expansion order.
- `class_num = 14`: number of diagnosis classes.
- `Acti_type_1 = 4`: activation in the polynomial approximation stage
  (`tanh`).
- `Acti_type_2 = 2`: activation in the voting stage (`LeakyReLU`).
- `diff_type = 2`: squared residual fault feature.
- `Gaus_type = 1`: no additional Gaussian/normalization transform.

Training prints per-epoch training and test accuracy for each fold, followed by
the averaged final metrics.

## Notes

- The code constructs labels automatically by assuming the same number of
  samples per class.
- The train/test normalization uses only the training split statistics for each
  fold and applies the same scaling to the corresponding test split.
- The implementation is intended for experimental reproduction and analysis, so
  the script includes additional metric and visualization code after the main
  training loop.
