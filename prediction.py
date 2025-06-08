import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration for file paths ---
DATA_PATHS = {
    'product_category_translation': 'product_category_name_translation.csv',
}

# --- Load the saved model and training columns ---
MODEL_PATH = 'random_forest_model.pkl'
TRAINING_COLUMNS_PATH = 'model_training_columns.pkl'

try:
    model = joblib.load(MODEL_PATH)
    training_columns = joblib.load(TRAINING_COLUMNS_PATH)
    print("Model and training columns loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading required files: {e}. Make sure '{MODEL_PATH}' and '{TRAINING_COLUMNS_PATH}' exist.")
    model = None
    training_columns = []
except Exception as e:
    print(f"An unexpected error occurred while loading files: {e}")
    model = None
    training_columns = []


# --- 2. Feature Engineering Function (simplified for single-row inference) ---
def create_features_for_inference(df_raw_input):
    """
    Creates engineered features for a single (or small batch of) raw input order data.
    Note: For aggregate features (seller_avg_review_score), this function assumes
    these are either present in df_raw_input or can be looked up from a pre-computed source.
    If not provided, NaNs will be present and handled by the model's imputer.

    Args:
        df_raw_input (pd.DataFrame): A DataFrame (typically one row) mimicking the
                                     structure of the original df_merged before feature engineering.

    Returns:
        pd.DataFrame: DataFrame with engineered features, ready for prediction.
    """
    df = df_raw_input.copy()

    # Convert relevant columns to datetime objects
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date',
        'review_creation_date', 'review_answer_timestamp', 'shipping_limit_date'
    ]
    for col in date_cols:
        # Use errors='coerce' to turn unparseable dates into NaT
        if col in df.columns and df[col].dtype == 'object': # Only convert if it's an object/string
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y %H:%M', errors='coerce')


    # Order-level Features (must be present in df_raw_input or derived)
    # Handle cases where datetime columns might be NaT
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['order_processing_time'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.days
    df['is_delivery_on_time'] = (df['delivery_delay'].fillna(0) <= 0).astype(int) # Handle NaNs for comparison
    df['is_review_commented'] = df['review_comment_message'].notna().astype(int)

    # Handle division by zero and missing values for price_per_weight
    df['price_per_weight'] = df['price'] / (df['product_weight_g'].fillna(1.0) + 1e-6) # Fill product_weight_g NaNs for calculation

    # has_multiple_items: For a single new order, this is typically 0 unless external logic implies otherwise
    # If `order_item_id` is present and represents count, then use that. Otherwise, default to 0 for a single new item.
    if 'order_item_id' in df.columns and len(df) > 1: # Assuming `order_item_id` in input means items within this transaction
         df['has_multiple_items'] = (df.groupby('order_id')['order_item_id'].transform('count') > 1).astype(int)
    else:
        df['has_multiple_items'] = 0

    df['payment_is_voucher'] = (df['payment_type'] == 'voucher').astype(int)
    df['payment_is_boleto'] = (df['payment_type'] == 'boleto').astype(int)
    # `is_canceled_order` is the target, so we don't derive it from order_status for inference input X
    # but it might be temporarily created if order_status is present and used for other derived features
    # in the create_features_for_inference function. For the final X passed to predict, it should be absent.
    # The current create_features in training calculates it, so if this inference func is a mirror,
    # it might create it, but then it's dropped before passing to the model.
    df['review_response_time'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.days


    # IMPORTANT: For single-row inference, aggregate features (like seller_avg_review_score)
    # cannot be calculated from the single row itself. They must be provided as part of the
    # df_raw_input or looked up from a pre-computed data store (e.g., a database of seller/customer aggregates).
    # If they are not passed, they will be NaN and the imputer in the pipeline will handle them.

    # Ensure these aggregate columns exist, even if with default NaNs, before returning
    # This prevents KeyError if user doesn't provide them via Streamlit inputs
    # List all the aggregate columns that were created in training
    agg_cols = [
        'seller_avg_delivery_delay', 'seller_avg_review_score', 'seller_cancellation_rate',
        'seller_total_sales_value', 'seller_avg_price', 'seller_avg_freight_value',
        'seller_review_comment_rate', 'seller_distinct_products', 'seller_order_approval_time_avg',
        'seller_max_delivery_delay', 'seller_min_delivery_delay', 'seller_num_orders',
        'customer_avg_review_score', 'customer_cancellation_rate', 'customer_total_spent',
        'customer_avg_payment_installments', 'customer_distinct_sellers', 'customer_distinct_products',
        'customer_review_comment_given_rate', 'customer_avg_delivery_delay', 'customer_num_orders'
    ]
    for col in agg_cols:
        if col not in df.columns:
            df[col] = np.nan # Or a sensible default if you have one

    return df

def make_prediction(raw_input_data: pd.DataFrame):
    """
    Makes a prediction using the loaded Random Forest model.

    Args:
        raw_input_data (pd.DataFrame): A DataFrame containing new raw data for prediction,
                                       mimicking the structure of the original merged data
                                       before feature engineering. This should typically be one row.

    Returns:
        tuple: Predicted labels (0 for not canceled, 1 for canceled) and probabilities.
    """
    if model is None:
        print("Model not loaded, cannot make prediction.")
        return np.array([]), np.array([])

    # 1. Apply feature engineering to the raw input data
    engineered_input_data = create_features_for_inference(raw_input_data)

    # Remove 'is_canceled_order' if it was created, as it's the target and should not be in X
    if 'is_canceled_order' in engineered_input_data.columns:
        engineered_input_data = engineered_input_data.drop(columns=['is_canceled_order'])

    # 2. Align columns with training data schema
    # Create a DataFrame with all training columns, filled with NaN initially
    final_input_df = pd.DataFrame(columns=training_columns)

    # Populate it with values from the engineered_input_data
    for col in final_input_df.columns:
        if col in engineered_input_data.columns:
            final_input_df[col] = engineered_input_data[col]
        else:
            final_input_df[col] = np.nan # Fill missing engineered features with NaN, imputer will handle

    # Ensure the order of columns matches the training order exactly
    final_input_df = final_input_df[training_columns]

    # 3. Make prediction
    predictions = model.predict(final_input_df)
    probabilities = model.predict_proba(final_input_df)[:, 1]
    return predictions, probabilities

if __name__ == '__main__':
    print("--- Testing Prediction Function ---")

    # --- INPUTS DESIGNED FOR HIGH RISK (EVEN MORE EXTREME) ---
    current_time = pd.Timestamp.now().strftime('%d-%m-%Y %H:%M')
    # Make actual delivery far in the past to ensure large delay
    past_delivery_time = (pd.Timestamp.now() - pd.Timedelta(days=120)).strftime('%d-%m-%Y %H:%M') # 4 months ago
    # Make estimated delivery recent to make the delay even larger
    estimated_delivery_time = (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime('%d-%m-%Y %H:%M')
    approved_at_time = (pd.Timestamp.now() - pd.Timedelta(days=115)).strftime('%d-%m-%Y %H:%M')
    carrier_date_time = (pd.Timestamp.now() - pd.Timedelta(days=110)).strftime('%d-%m-%Y %H:%M')
    review_creation_time = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%d-%m-%Y %H:%M')
    review_answer_time = (pd.Timestamp.now() - pd.Timedelta(days=50)).strftime('%d-%m-%Y %H:%M') # Long response time
    shipping_limit_time = (pd.Timestamp.now() - pd.Timedelta(days=118)).strftime('%d-%m-%Y %H:%M')


    dummy_raw_input_data = {
        'order_id': ['extreme_risk_order_001'],
        'customer_id': ['super_fraud_customer_xyz'],
        'customer_unique_id': ['ultimate_abuser_123'],
        'order_status': ['canceled'], # Explicitly setting status to canceled for this test input to reflect extreme case
        'order_purchase_timestamp': ['01-01-2023 10:00'], # Very old purchase
        'order_approved_at': [approved_at_time],
        'order_delivered_carrier_date': [carrier_date_time],
        'order_delivered_customer_date': [past_delivery_time], # Extremely delayed
        'order_estimated_delivery_date': [estimated_delivery_time],
        'review_creation_date': [review_creation_time],
        'review_answer_timestamp': [review_answer_time],
        'shipping_limit_date': [shipping_limit_time],
        'payment_sequential': [1],
        'payment_type': ['voucher'], # High risk payment type
        'payment_installments': [1],
        'payment_value': [5.0], # Extremely low total value
        'product_id': ['critical_bad_prod_id_001'],
        'seller_id': ['critical_risky_seller_id_001'],
        'price': [1.0], # Extremely low product price
        'freight_value': [50.0], # Freight massively higher than price
        'product_category_name': ['servicos'], # Could imply a service that's easy to dispute
        'product_name_lenght': [5], # Extremely short, suspicious product name
        'product_description_lenght': [10], # Extremely short, suspicious description
        'product_photos_qty': [0], # No product photos
        'product_weight_g': [1], # Extremely light product
        'product_length_cm': [1],
        'product_height_cm': [1],
        'product_width_cm': [1],
        'customer_zip_code_prefix': [99999], # Unusual zip code
        'customer_city': ['cidade_fantasma'], # Unusual city
        'customer_state': ['RR'], # Most remote/less common state
        'seller_zip_code_prefix': [11111], # Unusual zip code
        'seller_city': ['vila_sumida'], # Unusual city
        'seller_state': ['AP'], # Another remote/less common state
        'review_score': [1], # Lowest review score
        'review_comment_title': ['FRAUD ALERT!!!'],
        'review_comment_message': ['This is an absolute scam. Product never delivered, seller is a ghost. Demand refund!'],
        'order_item_id': [1], # Assuming single item

        # EVEN MORE EXTREME Simulated Aggregated Features
        'seller_avg_delivery_delay': [30.0], # Seller's average delay is extremely high
        'seller_avg_review_score': [1.0], # Seller's average review score is the absolute worst
        'seller_cancellation_rate': [0.95], # 95% of seller's orders are canceled
        'seller_total_sales_value': [100.0], # Extremely low total sales for seller
        'seller_avg_price': [5.0], # Very low average price for seller's products
        'seller_avg_freight_value': [30.0], # Very high average freight for seller
        'seller_review_comment_rate': [1.0], # All seller's orders have negative comments
        'seller_distinct_products': [1], # Seller offers only one product
        'seller_order_approval_time_avg': [10.0], # Seller takes extremely long to approve orders
        'seller_max_delivery_delay': [100.0],
        'seller_min_delivery_delay': [10.0],
        'seller_num_orders': [2], # Extremely low number of orders, but nearly all canceled
        'customer_avg_review_score': [1.0], # Customer consistently gives lowest reviews
        'customer_cancellation_rate': [1.0], # 100% of customer's orders are canceled
        'customer_total_spent': [10.0], # Customer has spent almost nothing
        'customer_avg_payment_installments': [1.0], # Always single payment
        'customer_distinct_sellers': [1], # Customer only buys from one seller (suspicious)
        'customer_distinct_products': [1], # Customer only buys one type of product (suspicious)
        'customer_review_comment_given_rate': [1.0], # All customer's reviews have comments (bad ones)
        'customer_avg_delivery_delay': [30.0], # Customer consistently experiences extreme delays
        'customer_num_orders': [2], # Very few orders, but all canceled
        'product_category_name_english': ['services'] # Use 'services' for the English translation for consistency
    }
    dummy_raw_df = pd.DataFrame(dummy_raw_input_data)


    predictions, probabilities = make_prediction(dummy_raw_df)
    print(f"Predictions: {predictions}")
    print(f"Probabilities of cancellation: {probabilities}")
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration for file paths ---
DATA_PATHS = {
    'product_category_translation': 'product_category_name_translation.csv',
}

# --- Load the saved model and training columns ---
MODEL_PATH = 'random_forest_model.pkl'
TRAINING_COLUMNS_PATH = 'model_training_columns.pkl'

try:
    model = joblib.load(MODEL_PATH)
    training_columns = joblib.load(TRAINING_COLUMNS_PATH)
    print("Model and training columns loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading required files: {e}. Make sure '{MODEL_PATH}' and '{TRAINING_COLUMNS_PATH}' exist.")
    model = None
    training_columns = []
except Exception as e:
    print(f"An unexpected error occurred while loading files: {e}")
    model = None
    training_columns = []


# --- 2. Feature Engineering Function (simplified for single-row inference) ---
def create_features_for_inference(df_raw_input):
    """
    Creates engineered features for a single (or small batch of) raw input order data.
    Note: For aggregate features (seller_avg_review_score), this function assumes
    these are either present in df_raw_input or can be looked up from a pre-computed source.
    If not provided, NaNs will be present and handled by the model's imputer.

    Args:
        df_raw_input (pd.DataFrame): A DataFrame (typically one row) mimicking the
                                     structure of the original df_merged before feature engineering.

    Returns:
        pd.DataFrame: DataFrame with engineered features, ready for prediction.
    """
    df = df_raw_input.copy()

    # Convert relevant columns to datetime objects
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date',
        'review_creation_date', 'review_answer_timestamp', 'shipping_limit_date'
    ]
    for col in date_cols:
        # Use errors='coerce' to turn unparseable dates into NaT
        if col in df.columns and df[col].dtype == 'object': # Only convert if it's an object/string
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y %H:%M', errors='coerce')


    # Order-level Features (must be present in df_raw_input or derived)
    # Handle cases where datetime columns might be NaT
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['order_processing_time'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.days
    df['is_delivery_on_time'] = (df['delivery_delay'].fillna(0) <= 0).astype(int) # Handle NaNs for comparison
    df['is_review_commented'] = df['review_comment_message'].notna().astype(int)

    # Handle division by zero and missing values for price_per_weight
    df['price_per_weight'] = df['price'] / (df['product_weight_g'].fillna(1.0) + 1e-6) # Fill product_weight_g NaNs for calculation

    # has_multiple_items: For a single new order, this is typically 0 unless external logic implies otherwise
    # If `order_item_id` is present and represents count, then use that. Otherwise, default to 0 for a single new item.
    if 'order_item_id' in df.columns and len(df) > 1: # Assuming `order_item_id` in input means items within this transaction
         df['has_multiple_items'] = (df.groupby('order_id')['order_item_id'].transform('count') > 1).astype(int)
    else:
        df['has_multiple_items'] = 0

    df['payment_is_voucher'] = (df['payment_type'] == 'voucher').astype(int)
    df['payment_is_boleto'] = (df['payment_type'] == 'boleto').astype(int)
    df['review_response_time'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.days


    # IMPORTANT: For single-row inference, aggregate features (like seller_avg_review_score)
    # cannot be calculated from the single row itself. They must be provided as part of the
    # df_raw_input or looked up from a pre-computed data store (e.g., a database of seller/customer aggregates).
    # If they are not passed, they will be NaN and the imputer in the pipeline will handle them.

    # Ensure these aggregate columns exist, even if with default NaNs, before returning
    # This prevents KeyError if user doesn't provide them via Streamlit inputs
    # List all the aggregate columns that were created in training
    agg_cols = [
        'seller_avg_delivery_delay', 'seller_avg_review_score', 'seller_cancellation_rate',
        'seller_total_sales_value', 'seller_avg_price', 'seller_avg_freight_value',
        'seller_review_comment_rate', 'seller_distinct_products', 'seller_order_approval_time_avg',
        'seller_max_delivery_delay', 'seller_min_delivery_delay', 'seller_num_orders',
        'customer_avg_review_score', 'customer_cancellation_rate', 'customer_total_spent',
        'customer_avg_payment_installments', 'customer_distinct_sellers', 'customer_distinct_products',
        'customer_review_comment_given_rate', 'customer_avg_delivery_delay', 'customer_num_orders'
    ]
    for col in agg_cols:
        if col not in df.columns:
            df[col] = np.nan # Or a sensible default if you have one

    return df

def make_prediction(raw_input_data: pd.DataFrame):
    """
    Makes a prediction using the loaded LightGBM model.

    Args:
        raw_input_data (pd.DataFrame): A DataFrame containing new raw data for prediction,
                                       mimicking the structure of the original merged data
                                       before feature engineering. This should typically be one row.

    Returns:
        tuple: Predicted labels (0 for not canceled, 1 for canceled) and probabilities.
    """
    if model is None:
        print("Model not loaded, cannot make prediction.")
        return np.array([]), np.array([])

    # 1. Apply feature engineering to the raw input data
    engineered_input_data = create_features_for_inference(raw_input_data)

    # Remove 'is_canceled_order' if it was created, as it's the target and should not be in X
    if 'is_canceled_order' in engineered_input_data.columns:
        engineered_input_data = engineered_input_data.drop(columns=['is_canceled_order'])

    # 2. Align columns with training data schema
    # Create a DataFrame with all training columns, filled with NaN initially
    final_input_df = pd.DataFrame(columns=training_columns)

    # Populate it with values from the engineered_input_data
    for col in final_input_df.columns:
        if col in engineered_input_data.columns:
            final_input_df[col] = engineered_input_data[col]
        else:
            final_input_df[col] = np.nan # Fill missing engineered features with NaN, imputer will handle

    # Ensure the order of columns matches the training order exactly
    final_input_df = final_input_df[training_columns]

    # 3. Make prediction
    predictions = model.predict(final_input_df)
    probabilities = model.predict_proba(final_input_df)[:, 1]
    return predictions, probabilities

if __name__ == '__main__':
    print("--- Testing Prediction Function ---")

    # --- INPUTS DESIGNED FOR HIGH RISK (MAXIMUM EXTREME) ---
    current_time = pd.Timestamp.now().strftime('%d-%m-%Y %H:%M')
    # Make actual delivery extremely far in the past to ensure massive delay
    past_delivery_time = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%d-%m-%Y %H:%M') # A year ago
    # Make estimated delivery very recent to make the delay difference enormous
    estimated_delivery_time = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime('%d-%m-%Y %H:%M')
    # Similarly extreme values for other timestamps
    approved_at_time = (pd.Timestamp.now() - pd.Timedelta(days=300)).strftime('%d-%m-%Y %H:%M')
    carrier_date_time = (pd.Timestamp.now() - pd.Timedelta(days=290)).strftime('%d-%m-%Y %H:%M')
    review_creation_time = (pd.Timestamp.now() - pd.Timedelta(days=280)).strftime('%d-%m-%Y %H:%M')
    review_answer_time = (pd.Timestamp.now() - pd.Timedelta(days=200)).strftime('%d-%m-%Y %H:%M')
    shipping_limit_time = (pd.Timestamp.now() - pd.Timedelta(days=350)).strftime('%d-%m-%Y %H:%M')


    dummy_raw_input_data = {
        'order_id': ['ultimate_risk_order_001'],
        'customer_id': ['ultimate_fraud_customer_xyz'],
        'customer_unique_id': ['final_abuser_123'],
        'order_status': ['canceled'], # Explicitly setting status to canceled for this test input to reflect extreme case
        'order_purchase_timestamp': ['01-01-2022 10:00'], # Very old purchase
        'order_approved_at': [approved_at_time],
        'order_delivered_carrier_date': [carrier_date_time],
        'order_delivered_customer_date': [past_delivery_time], # Extremely, massively delayed
        'order_estimated_delivery_date': [estimated_delivery_time],
        'review_creation_date': [review_creation_time],
        'review_answer_timestamp': [review_answer_time],
        'shipping_limit_date': [shipping_limit_time],
        'payment_sequential': [1],
        'payment_type': ['voucher'], # High risk payment type
        'payment_installments': [1],
        'payment_value': [1.0], # Extremely low total value
        'product_id': ['critical_bad_prod_id_001_v2'],
        'seller_id': ['critical_risky_seller_id_001_v2'],
        'price': [0.5], # Extremely low product price
        'freight_value': [100.0], # Freight astronomically higher than price
        'product_category_name': ['outros_acessorios'], # Could be a category with high dispute rates
        'product_name_lenght': [1], # Minimum length
        'product_description_lenght': [1], # Minimum length
        'product_photos_qty': [0], # Zero photos
        'product_weight_g': [1], # Minimum weight
        'product_length_cm': [1],
        'product_height_cm': [1],
        'product_width_cm': [1],
        'customer_zip_code_prefix': [00000], # Potentially invalid zip code
        'customer_city': ['nonexistent_city'], # Fictional city
        'customer_state': ['RR'], # Most remote/least common state
        'seller_zip_code_prefix': [00000], # Potentially invalid zip code
        'seller_city': ['fraud_city'], # Fictional city
        'seller_state': ['AM'], # Another remote/less common state
        'review_score': [1], # Lowest review score
        'review_comment_title': ['SCAM - DO NOT BUY!!!'],
        'review_comment_message': ['This is an absolute fraud. Product never delivered. Seller is a ghost. I will report this to authorities.'],
        'order_item_id': [1], # Assuming single item

        # Maximum Extreme Simulated Aggregated Features
        'seller_avg_delivery_delay': [60.0], # Seller's average delay is two months
        'seller_avg_review_score': [1.0], # Seller's average review score is the absolute worst
        'seller_cancellation_rate': [1.0], # 100% of seller's orders are canceled
        'seller_total_sales_value': [10.0], # Extremely low total sales for seller
        'seller_avg_price': [1.0], # Very low average price for seller's products
        'seller_avg_freight_value': [50.0], # Very high average freight for seller
        'seller_review_comment_rate': [1.0], # All seller's orders have negative comments
        'seller_distinct_products': [1], # Seller offers only one type of product
        'seller_order_approval_time_avg': [20.0], # Seller takes extremely long to approve orders
        'seller_max_delivery_delay': [365.0],
        'seller_min_delivery_delay': [30.0],
        'seller_num_orders': [2], # Minimal orders, all canceled
        'customer_avg_review_score': [1.0], # Customer consistently gives lowest reviews
        'customer_cancellation_rate': [1.0], # 100% of customer's orders are canceled
        'customer_total_spent': [1.0], # Customer has spent almost nothing
        'customer_avg_payment_installments': [1.0], # Always single payment
        'customer_distinct_sellers': [1], # Customer only buys from one seller (suspicious)
        'customer_distinct_products': [1], # Customer only buys one type of product (suspicious)
        'customer_review_comment_given_rate': [1.0], # All customer's reviews have comments (bad ones)
        'customer_avg_delivery_delay': [60.0], # Customer consistently experiences extreme delays
        'customer_num_orders': [2], # Minimal orders, all canceled
        'product_category_name_english': ['other_accessories'] # Use 'other_accessories' for the English translation for consistency
    }
    dummy_raw_df = pd.DataFrame(dummy_raw_input_data)


    predictions, probabilities = make_prediction(dummy_raw_df)
    print(f"Predictions: {predictions}")
    print(f"Probabilities of cancellation: {probabilities}")
