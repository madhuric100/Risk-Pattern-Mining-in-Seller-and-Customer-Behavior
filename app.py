import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Olist Fraud & Risk Dashboard")

# --- Configuration for file paths ---
DATA_PATHS = {
    'product_category_translation': 'product_category_name_translation.csv',
}

# --- Load Model and Training Columns ---
MODEL_PATH = 'random_forest_model.pkl'
TRAINING_COLUMNS_PATH = 'model_training_columns.pkl'

model = None
training_columns = []

try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TRAINING_COLUMNS_PATH):
        st.error(f"Required model files not found. Please ensure '{MODEL_PATH}' and '{TRAINING_COLUMNS_PATH}' are in the same directory.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    training_columns = joblib.load(TRAINING_COLUMNS_PATH)
    st.sidebar.success("Model and training schema loaded.")
except Exception as e:
    st.error(f"Failed to load the ML model or training schema: {e}")
    st.stop()

# --- Feature Engineering Function (re-used from predict_api.py) ---
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
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y %H:%M', errors='coerce')


    # Order-level Features (must be present in df_raw_input or derived)
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['order_processing_time'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.days
    df['is_delivery_on_time'] = (df['delivery_delay'].fillna(0) <= 0).astype(int)
    df['is_review_commented'] = df['review_comment_message'].notna().astype(int)
    df['price_per_weight'] = df['price'] / (df['product_weight_g'].fillna(1.0) + 1e-6)

    if 'order_item_id' in df.columns and len(df) > 1:
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

st.title("🛡️ Olist Fraud & Risk Prediction")
st.markdown("This dashboard predicts the likelihood of an order being canceled based on various order, seller, and customer features.")

st.sidebar.header("Input New Order Data")
st.sidebar.markdown("Adjust the parameters below to see the prediction for a hypothetical order.")

# --- Define the high-risk default values from predict_api.py ---
# Note: These values are chosen to be extreme and trigger a high risk prediction.
# Streamlit widgets have min/max limits, so we'll use the most extreme values allowed by those limits.
default_high_risk_inputs = {
    'price': 0.5,
    'freight_value': 100.0,
    'payment_type': 'voucher',
    'payment_installments': 1,
    'product_category_name_english': 'other_accessories', # Ensure this is in the selectbox list
    'product_weight_g': 1,
    'product_length_cm': 1,
    'product_height_cm': 1,
    'product_width_cm': 1,
    'product_name_lenght': 1,
    'product_description_lenght': 1,
    'product_photos_qty': 0,

    'delivery_delay_input': 30, # Max allowed in current Streamlit slider
    'order_processing_time_input': 15, # Max allowed
    'review_response_time_input': 10, # Max allowed

    'review_score': 1,
    'is_review_commented_input': True,

    'customer_city': 'nonexistent_city', # Fictional city
    'customer_state': 'RR', # Most remote/least common state
    'seller_city': 'fraud_city', # Fictional city
    'seller_state': 'AM', # Another remote/less common state

    'seller_avg_delivery_delay': 60.0,
    'seller_avg_review_score': 1.0,
    'seller_cancellation_rate': 1.0,
    'seller_num_orders': 2, # Minimal orders, all canceled
    'seller_distinct_products': 1,

    'customer_total_spent': 1.0,
    'customer_cancellation_rate': 1.0,
    'customer_num_orders': 2,
    'customer_avg_review_score': 1.0,
}


# --- Input Fields (Pre-populated with high-risk defaults) ---
st.sidebar.subheader("Order Details")
price = st.sidebar.number_input("Product Price (BRL)", min_value=0.0, value=float(default_high_risk_inputs['price']), step=0.1, key='price_input')
freight_value = st.sidebar.number_input("Freight Value (BRL)", min_value=0.0, value=float(default_high_risk_inputs['freight_value']), step=1.0, key='freight_input')
payment_type = st.sidebar.selectbox("Payment Type", ['credit_card', 'boleto', 'voucher', 'debit_card', 'not_defined'], index=['credit_card', 'boleto', 'voucher', 'debit_card', 'not_defined'].index(default_high_risk_inputs['payment_type']), key='payment_type_input')
payment_installments = st.sidebar.number_input("Payment Installments", min_value=1, value=int(default_high_risk_inputs['payment_installments']), step=1, key='installments_input')

# Ensure product_category_name_english list is comprehensive
all_product_categories = [
    'housewares', 'perfumery', 'auto', 'electronics', 'toys', 'watches_gifts',
    'telephony', 'fashion_bags_accessories', 'health_beauty', 'computers_accessories',
    'sports_leisure', 'bed_bath_table', 'furniture_decor', 'other',
    'agro_industry_and_commerce', 'air_conditioning', 'art', 'arts_and_craftsmanship',
    'audio', 'baby', 'books_general', 'books_imported', 'books_technical', 'christmas_gifts',
    'cinematography', 'construction_tools_construction', 'construction_tools_safety',
    'cool_stuff', 'costruction_tools_garden', 'costruction_tools_lights',
    'dvds_blu_ray', 'drinks', 'fashio_female_clothing', 'fashion_male_clothing',
    'fashion_childrens_clothes', 'fashion_formal_wear', 'fashion_underwear_beach',
    'fixed_telephony', 'flowers', 'food', 'food_drinks', 'furniture_bedroom',
    'furniture_living_room', 'furniture_mattress_and_upholstery', 'home_appliances',
    'home_appliances_2', 'home_comfort', 'home_comfort_2', 'home_construction',
    'house_and_video', 'industry_commerce_and_furniture', 'kitchen_dining_laundry_garden',
    'la_cuisine', 'luggage_accessories', 'market_place', 'musical_instruments',
    'music', 'office_furniture', 'party_articles', 'pet_shop', 'security_and_services',
    'signaling_and_security', 'small_appliances', 'small_appliances_home_security',
    'stationery', 'tablets_printing_image', 'the_arts', 'watches_gifts',
    'other_accessories', 'services' # Added 'services' and 'other_accessories'
]
product_category_name_english = st.sidebar.selectbox("Product Category (English)", all_product_categories, index=all_product_categories.index(default_high_risk_inputs['product_category_name_english']), key='category_input')

product_weight_g = st.sidebar.number_input("Product Weight (grams)", min_value=1, value=int(default_high_risk_inputs['product_weight_g']), step=1, key='weight_input')
product_length_cm = st.sidebar.number_input("Product Length (cm)", min_value=1, value=int(default_high_risk_inputs['product_length_cm']), step=1, key='length_input')
product_height_cm = st.sidebar.number_input("Product Height (cm)", min_value=1, value=int(default_high_risk_inputs['product_height_cm']), step=1, key='height_input')
product_width_cm = st.sidebar.number_input("Product Width (cm)", min_value=1, value=int(default_high_risk_inputs['product_width_cm']), step=1, key='width_input')
product_name_lenght = st.sidebar.number_input("Product Name Length", min_value=1, value=int(default_high_risk_inputs['product_name_lenght']), step=1, key='name_length_input')
product_description_lenght = st.sidebar.number_input("Product Description Length", min_value=1, value=int(default_high_risk_inputs['product_description_lenght']), step=1, key='desc_length_input')
product_photos_qty = st.sidebar.number_input("Product Photos Quantity", min_value=0, value=int(default_high_risk_inputs['product_photos_qty']), step=1, key='photos_qty_input')


st.sidebar.subheader("Timeliness Details (Simulated)")
# Increased max_value for more extreme inputs
delivery_delay_input = st.sidebar.number_input("Delivery Delay (Days, from est. delivery)", min_value=-100, max_value=365, value=int(default_high_risk_inputs['delivery_delay_input']), step=1, key='delivery_delay_input')
order_processing_time_input = st.sidebar.number_input("Order Approval Time (Days, from purchase)", min_value=0, max_value=30, value=int(default_high_risk_inputs['order_processing_time_input']), step=1, key='processing_time_input')
review_response_time_input = st.sidebar.number_input("Review Response Time (Days)", min_value=0, max_value=30, value=int(default_high_risk_inputs['review_response_time_input']), step=1, key='review_response_input')

st.sidebar.subheader("Review Details")
review_score = st.sidebar.slider("Review Score Given (1-5)", 1, 5, int(default_high_risk_inputs['review_score']), key='review_score_input')
is_review_commented_input = st.sidebar.checkbox("Review has a comment message?", value=default_high_risk_inputs['is_review_commented_input'], key='comment_checkbox')


st.sidebar.subheader("Customer & Seller Info (Simulated Aggregates)")
customer_city = st.sidebar.text_input("Customer City", default_high_risk_inputs['customer_city'], key='customer_city_input')
all_states = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
customer_state = st.sidebar.selectbox("Customer State", all_states, index=all_states.index(default_high_risk_inputs['customer_state']), key='customer_state_input')
seller_city = st.sidebar.text_input("Seller City", default_high_risk_inputs['seller_city'], key='seller_city_input')
seller_state = st.sidebar.selectbox("Seller State", all_states, index=all_states.index(default_high_risk_inputs['seller_state']), key='seller_state_input')


st.sidebar.markdown("**Seller Historical Metrics:**")
seller_avg_delivery_delay = st.sidebar.number_input("Avg Seller Delivery Delay (Days)", min_value=-50.0, max_value=100.0, value=float(default_high_risk_inputs['seller_avg_delivery_delay']), step=1.0, key='seller_avg_delay_input')
seller_avg_review_score = st.sidebar.slider("Avg Seller Review Score", 1.0, 5.0, float(default_high_risk_inputs['seller_avg_review_score']), format="%.1f", key='seller_avg_review_input')
seller_cancellation_rate = st.sidebar.slider("Seller Cancellation Rate", 0.0, 1.0, float(default_high_risk_inputs['seller_cancellation_rate']), format="%.2f", key='seller_cancel_rate_input')
seller_num_orders = st.sidebar.number_input("Seller Total Orders", min_value=1, value=int(default_high_risk_inputs['seller_num_orders']), step=1, key='seller_orders_input')
seller_distinct_products = st.sidebar.number_input("Seller Distinct Products", min_value=1, value=int(default_high_risk_inputs['seller_distinct_products']), step=1, key='seller_distinct_prod_input')


st.sidebar.markdown("**Customer Historical Metrics:**")
customer_total_spent = st.sidebar.number_input("Customer Total Spent (BRL)", min_value=0.0, max_value=100000.0, value=float(default_high_risk_inputs['customer_total_spent']), step=1.0, key='customer_spent_input')
customer_cancellation_rate = st.sidebar.slider("Customer Cancellation Rate", 0.0, 1.0, float(default_high_risk_inputs['customer_cancellation_rate']), format="%.2f", key='customer_cancel_rate_input')
customer_num_orders = st.sidebar.number_input("Customer Total Orders", min_value=1, value=int(default_high_risk_inputs['customer_num_orders']), step=1, key='customer_orders_input')
customer_avg_review_score = st.sidebar.slider("Avg Customer Review Score Given", 1.0, 5.0, float(default_high_risk_inputs['customer_avg_review_score']), format="%.1f", key='customer_avg_review_input')


if st.sidebar.button("Predict Order Risk"):
    if model:
        # Construct the raw input DataFrame for feature engineering
        # Use a consistent date format for parsing
        current_time_str = pd.Timestamp.now().strftime('%d-%m-%Y %H:%M')
        # Simulate extreme past dates for calculation of delays
        past_delivery_time_str = (pd.Timestamp.now() - pd.Timedelta(days=default_high_risk_inputs['delivery_delay_input'] + 5)).strftime('%d-%m-%Y %H:%M') # actual delivery
        estimated_delivery_time_str = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime('%d-%m-%Y %H:%M') # estimated delivery

        # Derive other time-related fields based on input delay/processing times
        purchase_time = pd.Timestamp.now() - pd.Timedelta(days=default_high_risk_inputs['order_processing_time_input'])
        approved_at_time = pd.Timestamp.now()
        carrier_date_time = pd.Timestamp.now() + pd.Timedelta(days=2) # 2 days after approval for carrier
        review_creation_time = pd.Timestamp.now() + pd.Timedelta(days=default_high_risk_inputs['review_response_time_input'] + 1)
        review_answer_time = pd.Timestamp.now() + pd.Timedelta(days=default_high_risk_inputs['review_response_time_input'])
        shipping_limit_time = pd.Timestamp.now() + pd.Timedelta(days=3) # Arbitrary, not directly controlled by input


        raw_input_data = {
            'order_id': ['simulated_order_1'],
            'customer_id': ['simulated_customer_1'],
            'customer_unique_id': ['simulated_unique_customer_1'],
            'order_status': ['processing'], # Placeholder for status, `is_canceled_order` will be derived
            'order_purchase_timestamp': [purchase_time.strftime('%d-%m-%Y %H:%M')],
            'order_approved_at': [approved_at_time.strftime('%d-%m-%Y %H:%M')],
            'order_delivered_carrier_date': [carrier_date_time.strftime('%d-%m-%Y %H:%M')],
            'order_delivered_customer_date': [past_delivery_time_str], # This is the actual delivery
            'order_estimated_delivery_date': [estimated_delivery_time_str], # This is the estimated delivery
            'review_creation_date': [review_creation_time.strftime('%d-%m-%Y %H:%M')],
            'review_answer_timestamp': [review_answer_time.strftime('%d-%m-%Y %H:%M')],
            'shipping_limit_date': [shipping_limit_time.strftime('%d-%m-%Y %H:%M')],

            'payment_sequential': [1],
            'payment_type': [payment_type],
            'payment_installments': [payment_installments],
            'payment_value': [price + freight_value], # Total value of the order
            'product_id': ['simulated_product_1'],
            'seller_id': ['simulated_seller_1'],
            'price': [price],
            'freight_value': [freight_value],
            'product_category_name': [product_category_name_english.replace('_', ' ')],
            'product_name_lenght': [product_name_lenght],
            'product_description_lenght': [product_description_lenght],
            'product_photos_qty': [product_photos_qty],
            'product_weight_g': [product_weight_g],
            'product_length_cm': [product_length_cm],
            'product_height_cm': [product_height_cm],
            'product_width_cm': [product_width_cm],

            'customer_zip_code_prefix': [00000], # Dummy
            'customer_city': [customer_city],
            'customer_state': [customer_state],
            'seller_zip_code_prefix': [00000], # Dummy
            'seller_city': [seller_city],
            'seller_state': [seller_state],

            'review_score': [review_score],
            'review_comment_title': [np.nan], # Keep as NaN, is_review_commented_input handles presence
            'review_comment_message': ["Some comment" if is_review_commented_input else np.nan],
            'order_item_id': [1], # Assuming one item per order for simplicity

            # --- Simulated Aggregated Features (from inputs) ---
            'seller_avg_delivery_delay': [seller_avg_delivery_delay],
            'seller_avg_review_score': [seller_avg_review_score],
            'seller_cancellation_rate': [seller_cancellation_rate],
            'seller_total_sales_value': [100000.0], # Dummy, can be adjusted
            'seller_avg_price': [120.0], # Dummy, can be adjusted
            'seller_avg_freight_value': [15.0], # Dummy, can be adjusted
            'seller_review_comment_rate': [0.5], # Dummy, can be adjusted
            'seller_distinct_products': [seller_distinct_products],
            'seller_order_approval_time_avg': [0.5], # Dummy, can be adjusted
            'seller_max_delivery_delay': [60.0], # Dummy
            'seller_min_delivery_delay': [-5.0], # Dummy
            'seller_num_orders': [seller_num_orders],

            'customer_avg_review_score': [customer_avg_review_score],
            'customer_cancellation_rate': [customer_cancellation_rate],
            'customer_total_spent': [customer_total_spent],
            'customer_avg_payment_installments': [1.0], # Dummy
            'customer_distinct_sellers': [5], # Dummy
            'customer_distinct_products': [8], # Dummy
            'customer_review_comment_given_rate': [0.6], # Dummy
            'customer_avg_delivery_delay': [0.0], # Dummy
            'customer_num_orders': [customer_num_orders]
        }
        input_df_raw = pd.DataFrame(raw_input_data)

        # 1. Apply feature engineering to the raw input data
        engineered_input_data = create_features_for_inference(input_df_raw)

        # Remove 'is_canceled_order' if it was created, as it's the target and should not be in X
        if 'is_canceled_order' in engineered_input_data.columns:
            engineered_input_data = engineered_input_data.drop(columns=['is_canceled_order'])

        # 2. Align columns with training data schema
        final_input_df = pd.DataFrame(columns=training_columns)
        for col in training_columns:
            if col in engineered_input_data.columns:
                final_input_df[col] = engineered_input_data[col]
            else:
                final_input_df[col] = np.nan # Fill missing engineered features with NaN

        # Ensure the order of columns matches the training order exactly
        final_input_df = final_input_df[training_columns] # Reorder to match training

        try:
            prediction = model.predict(final_input_df)[0]
            probability = model.predict_proba(final_input_df)[:, 1][0]

            st.subheader("Prediction Result")
            if probability > 0.7: # Using the 0.7 threshold
                st.markdown(f"🚨 **HIGH RISK**: This order is predicted to be CANCELED.")
                st.error(f"Probability of Cancellation: **{probability:.2f}**")
            else:
                st.markdown(f"✅ **LOW RISK**: This order is predicted to be DELIVERED successfully.")
                st.success(f"Probability of Cancellation: **{probability:.2f}**")

            st.markdown("---")
            st.subheader("How this prediction was made:")
            st.markdown("The model analyzed various features including payment details, delivery times, and simulated seller/customer historical behavior to assess the risk.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure all input values are valid and try again.")
    else:
        st.warning("Model not loaded. Cannot make predictions.")

st.markdown("---")
st.markdown("### About the Model")
st.info("""
This model uses a LightGBM Classifier to predict if an order will be canceled.
It leverages engineered features derived from various Olist datasets, including
time-based metrics, review sentiments, and aggregated seller/customer performance.
""")

st.markdown("---")

