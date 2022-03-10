# Predicting Customer Churn 
The data for this model is available on [Kaggle](https://www.kaggle.com/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) and contains data on genre, tenure, preferred payment mode, and more for over 5,600 customers of an e-commerce company.

### Use of:
* **Python** Version 3.9.7
* **Packages:** numpy, pandas, matplotlib, sklearn, picle, feature_engine, xgboost, missingno 

# Overview
* Analysed the data (distributions, type of data, missing values) and studied the relationship between the features and the average churn rate of an e-commerce
* Preprocessed the data to be able to apply a machine learning model
* Studied more than 10 classification algorithms, from simple models like Logistic Regression to ensemble methods like Random Forest or XGBoosting as well as combinations of models using Voting classifiers and Stacking classifiers
* Optimised the hyperparameters from the best performing models using 10-fold cross validation
* Created a pipeline able to preprocess and predict on new data
* Achieved high performance on test data such as:

| Best Model       |  Accuracy   | Precision   |  Recall    | 
| :--------------: | :---------: | :--------:  | :--------: | 
| XGBClassifier    |  0.00       |  0.00       |  0.00      |

<br><br>


![Confusion Matrix image](https://github.com/pcmaldonado/CustomerChurn/blob/main/conf_matrix.png)

<br>

----------------------------------------------------

### Note about metrics
* **Accuracy:** % of true pos/neg overall -it is a highly biased metric on imbalanced datasets
* **Precision:** % of correctly predicted positive cases from the total of positive predicted cases = (true positive / (true positive + false positive)
* **Recall:** % of positive outcomes correctly identified from the total number of positive cases = (true positive / (true positive + false negatives)

One way to get perfect **recall** is to classify every observation as a positive outcome. Indeed, this would catch all real positive outcomes. However, it would come with a lot of errors (false positives = Type I error), this would have a negative impact on the **precision** of the model that takes into account these errors. 

Moreover, trying to improve only the **precision** of the model could mean to avoid false positives at all costs, thus avoid predicting positive cases all together and mostly classify observations as negative outcomes, which would lead to a high number of false negatives (Type II Error), which would have a negative impact on the **recall**. 

Thus, there is a **trade-off** between recall and precision.



