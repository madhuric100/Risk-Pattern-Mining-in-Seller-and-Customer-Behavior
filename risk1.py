import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV # Changed from GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt

# --- Configuration for file paths ---
DATA_PATHS = {
    'sellers': 'olist_sellers_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'order_payments': 'olist_order_payments_dataset.csv',
    'geolocation': 'olist_geolocation_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'product_category_translation': 'product_category_name_translation.csv',
    'order_reviews': 'olist_order_reviews_dataset.csv',
    'orders': 'olist_orders_dataset.csv',
    'products': 'olist_products_dataset.csv'
}

# --- 1. Data Loading and Merging Function ---
def load_and_merge_data(data_paths):
    """
    Loads multiple Olist datasets and merges them into a single DataFrame.
    Args:
        data_paths (dict): A dictionary mapping dataset names to their CSV file paths.
    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    print("Loading and merging data...")
    df_olist_sellers = pd.read_csv(data_paths['sellers'])
    df_olist_customers = pd.read_csv(data_paths['customers'])
    df_olist_order_payments = pd.read_csv(data_paths['order_payments'])
    df_olist_order_items = pd.read_csv(data_paths['order_items'])
    df_product_category_name_translation = pd.read_csv(data_paths['product_category_translation'])
    df_olist_order_reviews = pd.read_csv(data_paths['order_reviews'])
    df_olist_orders = pd.read_csv(data_paths['orders'])
    df_olist_products = pd.read_csv(data_paths['products'])

    df_merged = pd.merge(df_olist_orders, df_olist_customers, on='customer_id', how='left')
    df_merged = pd.merge(df_merged, df_olist_order_reviews, on='order_id', how='left')
    df_merged = pd.merge(df_merged, df_olist_order_payments, on='order_id', how='left')
    df_merged = pd.merge(df_merged, df_olist_order_items, on='order_id', how='left')
    df_merged = pd.merge(df_merged, df_olist_products, on='product_id', how='left')
    df_merged = pd.merge(df_merged, df_olist_sellers, on='seller_id', how='left')
    df_merged = pd.merge(df_merged, df_product_category_name_translation, on='product_category_name', how='left')

    date_cols = [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date',
        'review_creation_date', 'review_answer_timestamp', 'shipping_limit_date'
    ]
    for col in date_cols:
        df_merged[col] = pd.to_datetime(df_merged[col], format='%d-%m-%Y %H:%M', errors='coerce')

    print("Data loaded and merged successfully.")
    return df_merged

# --- 2. Feature Engineering Function ---
def create_features(df_raw):
    """
    Creates new engineered features based on the raw merged DataFrame.
    Args:
        df_raw (pd.DataFrame): The raw merged DataFrame before feature engineering.
    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    df = df_raw.copy()

    print("Starting feature engineering...")

    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['order_processing_time'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.days
    df['is_delivery_on_time'] = (df['delivery_delay'] <= 0).astype(int)
    df['is_review_commented'] = df['review_comment_message'].notna().astype(int)
    df['price_per_weight'] = df['price'] / (df['product_weight_g'] + 1e-6)
    
    if 'order_item_id' in df.columns:
        order_item_counts = df.groupby('order_id')['order_item_id'].transform('count')
        df['has_multiple_items'] = (order_item_counts > 1).astype(int)
    else:
        df['has_multiple_items'] = 0

    df['payment_is_voucher'] = (df['payment_type'] == 'voucher').astype(int)
    df['payment_is_boleto'] = (df['payment_type'] == 'boleto').astype(int)
    df['is_canceled_order'] = (df['order_status'] == 'canceled').astype(int)
    df['review_response_time'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.days

    seller_features = df.groupby('seller_id').agg(
        seller_avg_delivery_delay=('delivery_delay', 'mean'),
        seller_avg_review_score=('review_score', 'mean'),
        seller_cancellation_rate=('is_canceled_order', 'mean'),
        seller_total_sales_value=('payment_value', 'sum'),
        seller_avg_price=('price', 'mean'),
        seller_avg_freight_value=('freight_value', 'mean'),
        seller_review_comment_rate=('is_review_commented', 'mean'),
        seller_distinct_products=('product_id', 'nunique'),
        seller_order_approval_time_avg=('order_processing_time', 'mean'),
        seller_max_delivery_delay=('delivery_delay', 'max'),
        seller_min_delivery_delay=('delivery_delay', 'min'),
        seller_num_orders=('order_id', 'nunique')
    ).reset_index()
    df = pd.merge(df, seller_features, on='seller_id', how='left')

    customer_features = df.groupby('customer_unique_id').agg(
        customer_avg_review_score=('review_score', 'mean'),
        customer_cancellation_rate=('is_canceled_order', 'mean'),
        customer_total_spent=('payment_value', 'sum'),
        customer_avg_payment_installments=('payment_installments', 'mean'),
        customer_distinct_sellers=('seller_id', 'nunique'),
        customer_distinct_products=('product_id', 'nunique'),
        customer_review_comment_given_rate=('is_review_commented', 'mean'),
        customer_avg_delivery_delay=('delivery_delay', 'mean'),
        customer_num_orders=('order_id', 'nunique')
    ).reset_index()
    df = pd.merge(df, customer_features, on='customer_unique_id', how='left')

    print("Feature engineering complete.")
    return df

# --- 3. Preprocessing and Model Training Function ---
def train_and_evaluate_model(df):
    """
    Preprocesses data, trains an LGBMClassifier with RandomizedSearchCV for tuning,
    and evaluates its performance. Also, provides threshold optimization.

    Args:
        df (pd.DataFrame): The DataFrame with engineered features.

    Returns:
        tuple: A tuple containing the best trained pipeline, X_test, y_test, and feature importances.
    """
    print("Preparing data for modeling...")
    y = df['is_canceled_order']

    features_to_exclude = [
        'order_id', 'customer_id', 'review_id', 'product_id', 'seller_id',
        'customer_unique_id', 'order_status',
        'is_canceled_order', # Correctly excluded target
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date',
        'review_creation_date', 'review_answer_timestamp', 'shipping_limit_date',
        'review_comment_title', 'review_comment_message',
        'product_category_name'
    ]

    X = df.drop(columns=[col for col in features_to_exclude if col in df.columns], errors='ignore')

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Calculate scale_pos_weight for LGBMClassifier
    neg_count = y.value_counts()[0]
    pos_count = y.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count
    print(f"Calculated initial scale_pos_weight: {scale_pos_weight_value:.2f}")


    # Define the pipeline for RandomizedSearchCV
    lgbm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(random_state=42, objective='binary'))
    ])

    # Define the parameter distribution for RandomizedSearchCV - Tuned for Precision
    # Using smaller ranges for faster execution, but still reasonable exploration
    param_distributions = {
        'classifier__n_estimators': [200, 300, 400],
        'classifier__learning_rate': [0.01, 0.02, 0.03],
        'classifier__num_leaves': [31, 63],
        'classifier__max_depth': [7, 10],
        'classifier__reg_alpha': [0.1, 0.5, 1.0],
        'classifier__reg_lambda': [0.1, 0.5, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9],
        'classifier__subsample': [0.8, 0.9],
        'classifier__min_child_samples': [30, 50, 100],
        'classifier__min_gain_to_split': [0.001, 0.01, 0.05],
        'classifier__scale_pos_weight': [scale_pos_weight_value, scale_pos_weight_value * 0.8, scale_pos_weight_value * 0.5]
    }

    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Starting RandomizedSearchCV for LGBMClassifier (this may take a while)...")
    random_search = RandomizedSearchCV( # Changed to RandomizedSearchCV
        lgbm_pipeline,
        param_distributions, # Changed to param_distributions
        n_iter=50, # Number of parameter settings that are sampled
        cv=3, # Using 3-fold cross-validation
        scoring='f1', # Optimize for F1-score of the positive class
        n_jobs=-1, # Use all available CPU cores
        random_state=42, # For reproducibility
        verbose=1
    )
    random_search.fit(X_train, y_train)

    best_model_pipeline = random_search.best_estimator_ # Changed to random_search
    print(f"\nBest parameters found by RandomizedSearchCV: {random_search.best_params_}") # Changed
    print(f"Best cross-validation F1-score: {random_search.best_score_:.4f}") # Changed

    print("\n--- Model Evaluation (Best Model) ---")
    y_pred = best_model_pipeline.predict(X_test)
    y_prob = best_model_pipeline.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # --- Threshold Optimization ---
    print("\n--- Threshold Optimization ---")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6) # Add epsilon to prevent division by zero
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    optimal_precision = precisions[optimal_threshold_idx]
    optimal_recall = recalls[optimal_threshold_idx]

    print(f"Optimal Threshold (F1-max): {optimal_threshold:.4f}")
    print(f"Precision at optimal threshold: {optimal_precision:.4f}")
    print(f"Recall at optimal threshold: {optimal_recall:.4f}")

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, marker='.', label='Precision-Recall curve')
    plt.scatter(optimal_recall, optimal_precision, color='red', s=100, label=f'Optimal (F1-max) Threshold: {optimal_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Canceled Orders')
    plt.legend()
    plt.grid(True)
    # plt.show() # Commented out as plt.show() might not display directly in some environments

    # Apply optimal threshold to test predictions
    y_pred_optimal_threshold = (y_prob >= optimal_threshold).astype(int)
    print(f"\n--- Model Evaluation with Optimal Threshold ({optimal_threshold:.4f}) ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_optimal_threshold))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_optimal_threshold))


    # --- Feature Importance (from the best trained classifier) ---
    all_feature_names = best_model_pipeline.named_steps['preprocessor'].get_feature_names_out()

    classifier = best_model_pipeline.named_steps['classifier']
    importances = classifier.feature_importances_

    feature_importances = pd.Series(importances, index=all_feature_names)
    print("\n--- Top 20 Feature Importances ---")
    print(feature_importances.nlargest(20))

    return best_model_pipeline, X.columns.tolist(), X_test, y_test, feature_importances

# --- Main execution flow ---
if __name__ == "__main__":
    df_merged = load_and_merge_data(DATA_PATHS)
    df_engineered = create_features(df_merged.copy())

    trained_pipeline, training_columns, X_test_eval, y_test_eval, feature_importances = train_and_evaluate_model(df_engineered)

    joblib.dump(trained_pipeline, 'lgbm_tuned_model.pkl')
    joblib.dump(training_columns, 'model_training_columns.pkl')
    print("\nModel saved as 'lgbm_tuned_model.pkl'")
    print("Training column names saved as 'model_training_columns.pkl'")

    feature_importances.to_csv('feature_importances.csv')
    print("Feature importances saved to 'feature_importances.csv'")
