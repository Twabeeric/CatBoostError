# CatBoost Errors

During the IEEE CIS Fraud competition, I spent alot of time debugging my Catboost model and I wanted to save anyone else the trouble.
I have a working example of my Catboost model uploaded but I have a quick and dirty summary of the errors below.
My model includes code for CatBoost Classification and feature selection. Feature selection does not lend itself well because the competition was judged on AUC score while binary labels were provided in the train data.
This makes it difficult to do feature selection as we have to decide on an arbitrary target border for the probabilities.

#Download the train/test data here
https://www.kaggle.com/c/ieee-fraud-detection/data

## Common Errors
### ValueError: could not convert string to float:
Separate your features into a list for numeric features and categorical features. Make sure no categorical features are listed as numeric features as catboost assumes anything not input as categorical is numeric.
Check your indices also, they are zero based indices and a cat_features indice should align with the correct indice for your feature names.


### Raise CatboostError("Invalid cat_features type={}: must be list() or np.ndarray()."
The dtype for your cat_features input must be a list or array

### CatboostError: Incorrect CD file. Invalid line number #10: catboost/libs/column_description/cd_parser.cpp:60: Invalid column index: index = 10, columnsCount = 10
Cat_features and feature_names are input as zero based indices usually with the label as the zero index (label=0)
For feature_names:

feature_names = dict()
for column, name in enumerate(train_df):
    if column == 0:
        continue
    feature_names1[column-1] = name     ###Notice (column-1)

For cat_features:
for i in catlist:
    cat_featuresls.append(train_df.columns.get_loc(i)+1)   ###Your cat_features index should begin at 1 

For feature selection:

features_to_evaluate = [0, 1, 2]  ###Notice that the first feature is index 0, the model skips the target label listed as 0.

### CatboostError: Invalid cat_features[55] = 451 value: must be < 446..
Check the length of your indices, length of indices should be one less than total number of features.

 ### CatBoostError: Error in dsv data. Line 1: Column 54 (type Num, value = "F"): catboost/libs/data_new/cb_dsv_loader.cpp:146: Factor 53 cannot be parsed as float. Try correcting column description file
Misaligned indices for cat_features leading to string cat_features being input as numeric features

### catboost/libs/metrics/metric.cpp:4643: All train targets are equal
By default, logloss considers the broder for classification to be 0.5, returning class labels 0 or 1.
You can set the target broder in parameters to be lower or higher than 0.5 to see what labela and accuracy is produces.

### CatBoostError Max target greater than 1: 3.57754e+06
Error with choosing the loss function, probabilities should not be greater than 1.

