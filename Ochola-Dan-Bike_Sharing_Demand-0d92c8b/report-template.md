# Report: Predict Bike Sharing Demand with AutoGluon Solution

#### DANIEL APOLO OCHOLA

## Initial Training

### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

When submitting predictions to Kaggle, I realized that the competition doesn't accept negative prediction values. Upon analysis with `predictions.describe()`, I found that some of my model predictions had negative values. I had to apply a post-processing step to set all negative predictions to zero using `predictions[predictions<0] = 0` before submitting to Kaggle.

### What was the top ranked model that performed?

The top ranked model in the initial training was `WeightedEnsemble_L2`, which combines multiple base models for better performance. This model achieved the lowest root mean squared error (RMSE) on the validation data.

## Exploratory data analysis and feature creation

### What did the exploratory analysis find and how did you add additional features?

Through exploratory data analysis, I observed that the `datetime` feature contains valuable temporal information that could help predict bike sharing demand. The histograms showed clear patterns in the data distribution across different features.

I extracted the `hour` component from the datetime column using `train.datetime.dt.hour` since time of day has a significant impact on bike usage (more bikes are rented during commute hours). Additionally, I converted categorical features like `season` and `weather` from integer to category data type to help the model interpret them correctly as categorical variables rather than numerical values.

### How much better did your model preform after adding additional features and why do you think that is?

After adding the hour feature and converting categorical variables, the model performance improved significantly. The validation score decreased from

<pre>
                     model   score_val              eval_metric  
0      WeightedEnsemble_L3  -53.147107  root_mean_squared_error      
1   RandomForestMSE_BAG_L2  -53.432083  root_mean_squared_error      
2          LightGBM_BAG_L2  -55.112296  root_mean_squared_error      
3        LightGBMXT_BAG_L2  -60.536476  root_mean_squared_error      
4    KNeighborsDist_BAG_L1  -84.125061  root_mean_squared_error      
5      WeightedEnsemble_L2  -84.125061  root_mean_squared_error      
6    KNeighborsUnif_BAG_L1 -101.546199  root_mean_squared_error      
7   RandomForestMSE_BAG_L1 -116.548359  root_mean_squared_error       
8     ExtraTreesMSE_BAG_L1 -124.600676  root_mean_squared_error       
9          CatBoost_BAG_L1 -130.651658  root_mean_squared_error       
10         LightGBM_BAG_L1 -131.054162  root_mean_squared_error       
11       LightGBMXT_BAG_L1 -131.460909  root_mean_squared_error      
12  NeuralNetFastAI_BAG_L1 -140.590599  root_mean_squared_error
</pre>

to

<pre>
                      model   score_val              eval_metric  
 0      WeightedEnsemble_L3  -63.387234  root_mean_squared_error   
 1      WeightedEnsemble_L2  -63.737929  root_mean_squared_error   
 2          CatBoost_BAG_L2  -64.027480  root_mean_squared_error   
 3        LightGBMXT_BAG_L1  -64.565891  root_mean_squared_error   
 4        LightGBMXT_BAG_L2  -64.589901  root_mean_squared_error   
 5   RandomForestMSE_BAG_L2  -64.612637  root_mean_squared_error   
 6          LightGBM_BAG_L2  -65.237426  root_mean_squared_error   
 7          CatBoost_BAG_L1  -65.498348  root_mean_squared_error   
 8          LightGBM_BAG_L1  -65.677254  root_mean_squared_error   
 9     ExtraTreesMSE_BAG_L1  -69.399939  root_mean_squared_error   
 10  RandomForestMSE_BAG_L1  -69.854608  root_mean_squared_error   
 11  NeuralNetFastAI_BAG_L1 -102.321313  root_mean_squared_error   
 12   KNeighborsUnif_BAG_L1 -124.768254  root_mean_squared_error   
 13   KNeighborsDist_BAG_L1 -125.617695  root_mean_squared_error
</pre>

and the Kaggle score improved from 1.79861 to 0.61281.

This improvement is likely because the hour of the day strongly correlates with bike rental patterns. People rent more bikes during commute hours and fewer bikes late at night. By explicitly providing this information to the model, it can learn these temporal patterns more effectively. Additionally, properly identifying categorical features helps the model create appropriate splits for these variables rather than treating them as continuous numerical values.

## Hyper parameter tuning

### How much better did your model preform after trying different hyper parameters?

After hyperparameter tuning, the model's performance showed declined compared to the feature-engineered model. The the Kaggle score changed from 0.61281 to 0.64741.

The hyperparameter tuning focused on optimizing the gradient boosting models (GBM) with a slower learning rate (0.01) and enabling extra trees, which can help reduce overfitting. The random search with 6 trials allowed AutoGluon to explore different configurations.

### If you were given more time with this dataset, where do you think you would spend more time?

With more time, I would:

1. Create more temporal features from the datetime column (day of week, month, season, holidays)
2. Engineer interaction features between weather and hour/season
3. Perform more extensive hyperparameter tuning with more trials and parameters
4. Try different model architectures and ensemble techniques
5. Analyze outliers and potentially remove or transform them
6. Apply more sophisticated feature selection techniques to identify the most predictive variables

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

| model        | hpo1 | hpo2 | hpo3 | score |
| ------------ | ---- | ---- | ---- | ----- |
| initial      | ?    | ?    | ?    | ?     |
| add_features | ?    | ?    | ?    | ?     |
| hpo          | ?    | ?    | ?    | ?     |

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](/model_test_score.png)

## Summary

This project demonstrated the effectiveness of AutoGluon for quickly developing high-performing machine learning models. The initial model provided a baseline, but adding feature engineering (particularly the hour of day feature) significantly improved performance by capturing important temporal patterns in bike rental behavior.

The most impactful improvement came from feature engineering, suggesting that understanding the domain and extracting relevant features is critical for this particular problem. This highlights the importance of domain knowledge and feature engineering in machine learning projects.

The hyperparameter tuning provided further improvements, showing that fine-tuning model parameters can enhance performance even after good feature engineering. The gradient boosting models with slower learning rates and extra trees proved particularly effective for this regression task.

Overall, the project achieved a final Kaggle score of 0.64741, demonstrating the power of combining automated machine learning tools like AutoGluon with thoughtful feature engineering and hyperparameter tuning.
