## Test-set metrics (paste into README)

| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.9294 | 0.9702 |      0.9209 |   0.8981 | 0.9093 | 0.8518 |
| Decision Tree       |     0.899  | 0.9293 |      0.8994 |   0.8375 | 0.8673 | 0.7873 |
| KNN                 |     0.9088 | 0.9602 |      0.893  |   0.8733 | 0.883  | 0.8084 |
| Naive Bayes         |     0.8339 | 0.9449 |      0.7178 |   0.9532 | 0.8189 | 0.6941 |
| SVM                 |     0.9283 | 0.9691 |      0.9304 |   0.8843 | 0.9068 | 0.8494 |

## Train vs test accuracy (use for overfitting observations)

|                     |   Train accuracy |   Test accuracy |     Gap |
|:--------------------|-----------------:|----------------:|--------:|
| Logistic Regression |           0.9307 |          0.9294 |  0.0013 |
| Decision Tree       |           0.9429 |          0.899  |  0.0439 |
| KNN                 |           0.9239 |          0.9088 |  0.0151 |
| Naive Bayes         |           0.8226 |          0.8339 | -0.0113 |
| SVM                 |           0.9546 |          0.9283 |  0.0263 |

Best by MCC: Logistic Regression | Best by AUC: Logistic Regression | Best by F1: Logistic Regression
