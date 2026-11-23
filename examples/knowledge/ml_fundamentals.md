# Machine Learning Fundamentals

## Bias and variance

The bias–variance trade-off describes two sources of prediction error. High bias means the model is too simple and under-fits: it misses real patterns in both the training and test data. High variance means the model is too flexible and over-fits: it memorises noise in the training data and performs much worse on new data. Regularisation, more training data, and ensembling reduce variance, while richer features and more expressive models reduce bias.

## Cross-validation

K-fold cross-validation splits the training data into K folds, trains on K−1 folds and validates on the remaining fold, repeating K times. The mean score estimates generalisation performance and the standard deviation shows how stable the model is. For classification, stratified K-fold keeps the class proportions identical in every fold, which matters for imbalanced data. A final untouched hold-out test set should still be kept to report an unbiased number.

## Evaluation metrics for classification

Accuracy is misleading on imbalanced data: predicting the majority class for a 95/5 split already gives 95% accuracy. Precision measures how many predicted positives are correct, recall measures how many true positives are found, and the F1 score is their harmonic mean. Macro-averaged F1 treats every class equally, which is why it is a good default for imbalanced problems. ROC-AUC measures ranking quality across all thresholds and equals the probability that a random positive is scored above a random negative.

## Evaluation metrics for regression

The coefficient of determination R² is the share of target variance explained by the model; 1.0 is perfect and 0 equals predicting the mean. RMSE penalises large errors more strongly than MAE because errors are squared before averaging. MAPE expresses the error as a percentage but is unstable when targets are close to zero.

## Gradient boosting

Gradient boosting builds an ensemble of shallow decision trees sequentially, where each new tree fits the residual errors of the ensemble so far. The learning rate shrinks each tree's contribution; smaller learning rates need more trees but usually generalise better. Histogram-based gradient boosting bins continuous features into at most 256 buckets, which makes training on large tabular datasets very fast and handles missing values natively.

## Feature engineering

One-hot encoding turns a categorical column into binary indicator columns and works well for low-cardinality categories. Identifier columns such as customer IDs must be dropped because they carry no generalisable signal and invite over-fitting. Skewed positive targets such as prices are often modelled on a log scale (log1p) and transformed back for prediction. Data leakage happens when a feature contains information that would not be available at prediction time, for example a column computed from the target.
