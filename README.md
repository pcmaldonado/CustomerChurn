# Predicting Customer Churn 
The data for this model is available on [Kaggle](https://www.kaggle.com/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) under the ["CC BY-NC-SA 4.0"](https://creativecommons.org/licenses/by-nc-sa/4.0/) License, and contains data on genre, tenure, preferred payment mode, and more for over 5,600 customers of an e-commerce company.

### Use of:
* **Python** Version 3.9.7
* **Packages:** numpy, pandas, matplotlib, sklearn, pickle, feature_engine, xgboost, missingno, smote, lime 

# Overview

## Main Results
<ul>
	<li>When <strong>tenure</strong> is low, there have been <strong>complains</strong> and people are <strong>single</strong>, they are highly likely to churn </li>
	<li>Some <strong>city tiers</strong> are also more likely to churn than others</li>
	<li>People who mostly buy from the <strong>Laptop & Accessory</strong> category are less likely to churn</li>
</ul>

The company should try to better focus on single people, study why certain categories are associated with a higher risk of churn,
as well as see why certain city tiers are more susceptible to churn. In addition, understanding where complaints are coming from could also be helpful.

By adressing these factors, it is probably that churn would decrease.</p>

<br>

## Walkthrough
<ul>
	<li>Analysed the data (distributions, type of data, missing values) and studied the relationship between the features and the average churn rate of the e-commerce</li>
	<li>Preprocessed the data to be able to apply a machine learning model</li>
	<li>Used SMOTE to handle imbalanced data</li>
	<li>Studied more than 10 classification algorithms, from simple models like Logistic Regression to ensemble methods like Random Forest or XGBoosting as well as combinations of models using Voting classifiers and Stacking classifiers</li>
	<li>Optimised the hyperparameters from the best performing models using 10-fold cross validation</li>
	<li>Combined the finetuned models using a Voting Classifier and a Stacking Classifier</li>
	<li>Created a pipeline able to preprocess and predict on new data using a pickled model</li>
	<li>Achieved higher performance on test data using the Voting Classifier on finetuned models, than using the best individual model:</li>
		<table>
			<tr>
				<th>Model</th>
				<th>Accuracy</th>
				<th>Precision</th>
				<th>Recall</th>
			</tr>
			<tr>
				<td>Voting Classifier (best model)</td>
				<td>97.96%</td>
				<td>96.02%</td>
				<td>91.35%</td>
			</tr>
			<tr>
			  <td>Extra Trees Classifier <br>(best individual model)</td>
			  <td>96.63%</td>
			  <td>96.82%</td>
			  <td>82.16%</td>
		  </tr>
		</table>
	<li>Used LIME to interpret results</li>
</ul>
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



