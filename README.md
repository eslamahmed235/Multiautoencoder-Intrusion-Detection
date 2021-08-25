# Intrusion-Detection-NSL-KDD-using-Autoencoders

### Note: Currently the implementation is in a notebook, due to time contraints. Working on converting the notebook to scripts and modularizing the code.

This project aims to detect Network Intrusion of the forms Denial of Service (DoS), Probe, User to Root(U2R), and Remote to Local (R2L) using an Autoencoder + ANN Classifier model. The dataset used is [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) by University of New Brunswick.

You can run this notebook [here](https://www.kaggle.com/ma1var3/nsl-kdd-classification-using-autoencoders)

## Dataset

The done analysis done by Gerry Saporito in the article ["A Deeper Dive into the NSL-KDD Data Set"](https://towardsdatascience.com/a-deeper-dive-into-the-nsl-kdd-data-set-15c753364657) , gives some insights about the structure and semantics of the dataset. The dataset has:

- 4 Categorical
- 6 Binary
- 23 Discrete
- 10 Continuous

The EDA done on [this](https://www.kaggle.com/stefanost/cnns-for-intrusion-detection) Kaggle kernel gives insights about the distribution of variables and the correlation of the features.

## Highlights

### Custom Loss Function

A custom loss function was used which is the hybrid of MSE and KL Divergence Loss, (work in progrss, yet to tune)

### Learning Rate Scheduling

An exponential learning rate decay function was used:

learning_rate = initial_learning_rate \* (drop ^ mfloor((1+epoch)/epoch_interval) )

## Results

| Encoding Dimension | Accuracy | Precision | AUC    | F1 Score |
| ------------------ | -------- | --------- | ------ | -------- |
| 10                 | 0.9804   | 0.9814    | 0.9989 | 0.9802   |
| 12                 | 0.9799   | 0.9803    | 0.9989 | 0.9799   |
| 14                 | 0.9833   | 0.9846    | 0.9990 | 0.9827   |
| 16                 | 0.9818   | 0.9828    | 0.9991 | 0.9819   |
| 18                 | 0.9813   | 0.9823    | 0.9989 | 0.9812   |
| 20                 | 0.9744   | 0.9768    | 0.9991 | 0.9744   |
| 22                 | 0.9819   | 0.9832    | 0.9989 | 0.9822   |
| 24                 | 0.9819   | 0.9833    | 0.9990 | 0.9822   |

## TO DO

- Modularising the code
- Combined Binary + Multiclass Classification
- Experimenting with custom layer (lambda layer for outputting absolute values)
- Experimenting with Conv1D layers for the classifier
- Multiautoencoder approach
