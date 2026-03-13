# CUSTOMER CHURN PREDICTION

This project demonstrates a **customer churn prediction** system. The main objective is to predict whether a customer is likely to churn based on their service usage and account information. Data and models are stored in **S3** for cloud-based storage and access.

## Models Used

* Random Forest
* XGBoost

## Model Test Results Comparison

| Metric                      | Random Forest | XGBoost |
|-----------------------------|---------------|---------|
| ROC AUC                     | 0.843         | 0.846   |
| Accuracy                    | 0.78          | 0.78    |
| Precision (Churn, class 1)  | 0.56          | 0.56    |
| Recall (Churn, class 1)     | 0.73          | 0.74    |
| F1-score (Churn, class 1)   | 0.64          | 0.64    |
| Confusion Matrix            | [821, 214]    | [822, 213]
                              | [100, 274]    | [99, 275]
                                                

## Model Selection

Random Forest and xgboost both shows near identical results. Random forest chosen as easier  interpretability and maintainable.

## Optimization & Scaling

### Scaling to 100k+ Records

The models can efficiently handle 100k+ records using:

* **Batch inference** - Process data in batches.
* **Parallel processing** - Both Random Forest and XGBoost leverage multi-threading for faster predictions.
* **Use cloud compute scaling** - Deploy on cloud instances.
* **Use Spark for data processing** - Store datasets in s3 and use spark for transformations instead of pandas.

### Retraining

* Store versioned datasets in S3
* Automate preprocessing with Airflow
* Validate schema before train again
* Use cloud pipelines with AWS Lambda, Step Functions, with Airflow.
* Save retrained models with version enabled in S3.

### Monitoring

* **Prediction monitoring** - As real world data changes over time, track output probabilities and feature distributions to detect data drift.
* **Performance monitoring** - Metrics such as accuracy, precision, recall.
* **Alerts** - Set up notifications if model performance falls below thresholds using CloudWatch.
* **ML Flow and DVC** - MLflow will remember every model training run. Git tracks code and small files, DVC tracks large data and models, remote storage holds the heavy stuff. In case of failure, a rollback can restore a previous stable model.

### Cost Considerations

* **Use serverless options when posible**: AWS Lambda, SageMaker endpoints reduce always-on server costs.
* **Storage**: Use S3 for datasets and model artifacts to minimize costs.
* **Compute**: Select appropriate instance types based on load. Consider auto-scaling to reduce idle resources.
* **File types**: Use proper file types to reduce cost