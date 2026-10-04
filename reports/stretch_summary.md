## CV variants (paired vs submitted recipe)

| variant          |   folds |   macro_f1 |   baseline_same_folds |   mean_paired_diff | folds_better   |   non_en_acc |   baseline_non_en_acc |   other_f1 |   baseline_other_f1 | verdict                  |
|:-----------------|--------:|-----------:|----------------------:|-------------------:|:---------------|-------------:|----------------------:|-----------:|--------------------:|:-------------------------|
| aug-idswap       |       5 |      0.893 |                 0.92  |             -0.027 | 0/5            |        0.923 |                 0.942 |      0.894 |               0.939 | hurts                    |
| aug-noise        |       5 |      0.905 |                 0.92  |             -0.015 | 2/5            |        0.942 |                 0.942 |      0.917 |               0.939 | hurts                    |
| aug-translate-f3 |       3 |      0.928 |                 0.929 |             -0.001 | 2/3            |        0.938 |                 0.938 |      0.897 |               0.933 | no detectable difference |
| hier0.5          |       5 |      0.905 |                 0.92  |             -0.014 | 1/5            |        0.942 |                 0.942 |      0.894 |               0.939 | hurts                    |
| hier1.0          |       5 |      0.902 |                 0.92  |             -0.019 | 0/5            |        0.923 |                 0.942 |      0.87  |               0.939 | hurts                    |
| oe               |       5 |      0.913 |                 0.92  |             -0.006 | 2/5            |        0.942 |                 0.942 |      0.894 |               0.939 | no detectable difference |

## Track B with / without Outlier Exposure

| run             | scorer            |   AUROC |   test_retention |   rejection_recall |   known_macro_f1 |
|:----------------|:------------------|--------:|-----------------:|-------------------:|-----------------:|
| model_b         | MSP (baseline)    |   0.791 |            0.938 |              0.262 |            0.941 |
| model_b         | energy            |   0.811 |            0.923 |              0.425 |            0.941 |
| model_b         | Mahalanobis [cls] |   0.82  |            0.908 |              0.338 |            0.941 |
| model_b         | kNN k=5 [cls]     |   0.813 |            0.923 |              0.288 |            0.941 |
| model_b_oe      | MSP (baseline)    |   0.776 |            0.908 |              0.412 |            0.944 |
| model_b_oe      | energy            |   0.781 |            0.954 |              0.375 |            0.944 |
| model_b_oe      | Mahalanobis [cls] |   0.845 |            0.908 |              0.425 |            0.944 |
| model_b_oe      | kNN k=5 [cls]     |   0.84  |            0.908 |              0.45  |            0.944 |
| model_b_near    | MSP (baseline)    |   0.744 |            0.958 |              0.143 |            0.903 |
| model_b_near    | energy            |   0.804 |            0.931 |              0.2   |            0.903 |
| model_b_near    | Mahalanobis [cls] |   0.854 |            0.889 |              0.343 |            0.903 |
| model_b_near    | kNN k=5 [cls]     |   0.8   |            0.931 |              0.171 |            0.903 |
| model_b_near_oe | MSP (baseline)    |   0.873 |            0.972 |              0.486 |            0.926 |
| model_b_near_oe | energy            |   0.893 |            0.931 |              0.486 |            0.926 |
| model_b_near_oe | Mahalanobis [cls] |   0.863 |            0.917 |              0.371 |            0.926 |
| model_b_near_oe | kNN k=5 [cls]     |   0.861 |            0.972 |              0.286 |            0.926 |