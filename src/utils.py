import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve, auc)


def missing_values(df, threshold=0):
    """
        Calculate the missing (NaN) values and their percentage in the given DataFrame.

        Args:
        - df (pd.DataFrame): The DataFrame to be examined.
        - threshold (float, optional): Threshold percentage for missing values.
            Default is 0.

        Returns:
        - pd.DataFrame: A DataFrame containing the count and percentage of missing values.
            If a threshold is specified, it includes missing values with a percentage below
            the specified threshold.
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round((100 * df.isnull().sum()/df.isnull().count()).sort_values(ascending=False), 2)
    df_missing_data = pd.concat([total, percent], axis=1, keys=['Count', 'Percent']).sort_values(by='Percent', ascending=False)
    
    if threshold == 0:
        return df_missing_data
    else:
        return df_missing_data[df_missing_data['Percent']<=threshold]


def detect_cardinality(df):
    """
        Detects the cardinality (number of unique values) for each feature in the given DataFrame.

        Args:
        - df (pd.DataFrame): The DataFrame for which cardinality is to be calculated.

        Returns:
        - pd.DataFrame: A DataFrame containing the cardinality count for each feature.
    """
    feature_cardinality = {}
        
    for column in df.columns:
        cardinality_value = df[column].nunique()
        feature_cardinality[column] = cardinality_value
    df_card_data = pd.DataFrame.from_dict(data=feature_cardinality, orient='index', columns=['Count'])
        
    return df_card_data


def evaluate_classification(y_true, y_pred):
    """
        Calculate Classification Metrics
        
        Args:
        y_true: True Value
        y_pred: Predict Value
        
        Returns:
        results: Dict Values of each metric 
    """
    results = {}

    # Accuracy
    results['Accuracy'] = accuracy_score(y_true, y_pred)

    # Precision
    results['Precision'] = precision_score(y_true, y_pred)

    # Recall
    results['Recall'] = recall_score(y_true, y_pred)

    # F1 Score
    results['F1 Score'] = f1_score(y_true, y_pred)

    # Confusion Matrix
    results['Confusion Matrix'] = confusion_matrix(y_true, y_pred)

    # fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # results['fpr'] = fpr
    # results['tpr'] = tpr
    # results['threshold'] = thresholds
    # results['Area Under Curve (AUC)'] = auc(fpr, tpr)

    return results