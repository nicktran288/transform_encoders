# Transform Encoders

Classes to perform encoding on categorical features.  Includes integer encoding and one-hot encoding. 


### Contents

- [General Information](#general-info)
- [Features](#features)
- [Technologies](#tech)
- [License](#license)


### General Information
Developed to streamline transforming features directly on pandas dataframes.  Specifically addresses certain obstacles encountered with standard implementations:
- intencoder: sklearn LabelEncoder does not work with null values
- ohencoder: built on pandas.get_dummies because sklearn OneHotEncoder was too complex for the pipeline at the time


### Features

##### intenecoder
- Identifies and transforms all object columns by default
- Automates sklearn LabelEncoder (inverse_)transform over arbitrary number of columns

##### ohencoder:
- Enables inverse_transform for pandas.get_dummies 


### Technologies
Built with Python 3.7

##### Uses the following packages:
- numpy
- pandas
- sklearn


### License
MIT 2019