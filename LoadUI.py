import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
# from src.L2_del import insert_file_into_collection, create_indexes
from L3_LoadElasticCloud import index_mongodb_data
from L2_PreprocMongo import insert_file_into_collection_raw
import yaml

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access MongoDB credentials from the YAML file
mongo_uri = config['mongodb']['uri']

# Access ECS credentials from the YAML file
cloud_id = config['ecs']['cloud_id']
api_key = config['ecs']['api_key']

# Set page configuration
st.set_page_config(
    page_title="Amazon Seller Price Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create session state for managing pages
if 'page' not in st.session_state:
    st.session_state.page = 'search'
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

def es_conn():
    # Connect to Elastic Cloud
    es = Elasticsearch(
        cloud_id=cloud_id,
        api_key=api_key
    )
    index_name = "data_ecom"
    return es, index_name

def fetch_data_from_es():
    try: 
        es, index_name = es_conn()
        # Use the scan helper to fetch all data from the index
        query = {"query": {"match_all": {}}}  # Modify query as needed
        results = scan(es, query=query, index=index_name, scroll='2m')
        # Convert the results to a DataFrame
        data = []
        for hit in results:
            data.append(hit['_source'])  # Only take the '_source' part of the hit
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        

# Function to load data
@st.cache_data
def data_load():
    try:
        
        df = fetch_data_from_es()
        
        # Ensure numeric columns are properly formatted
        numeric_cols = ['ratings', 'no_of_ratings', 'discount_price', 'actual_price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure historical price columns are numeric too
        hist_price_cols = [col for col in df.columns if col.startswith('discounted_price_')]
        for col in hist_price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Function to load the price prediction model
@st.cache_resource
def load_model():
    model = None
    try:
        with open('best_rf_price_recommender.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.sidebar.warning("Model file not found.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
    return None

# Improved function to predict optimal price for new product with sanity checks
def predict_optimal_price(model, product_features, df):
    if model is None:
        return None
    
    try:
        # Using the features from your model training
        features = [
            'price_difference', 'discount_percentage', 'category_popularity',
            'ratings_scaled', 'log_no_of_ratings', 'name_length',
            'main_category_encoded', 'sub_category_encoded'
        ]
        
        # Map to available features in your data
        feature_mapping = {
            'price_difference': 'price_difference',
            'discount_percentage': 'discount_percentage', 
            'category_popularity': 'popularity_score',  # Using as proxy
            'ratings_scaled': 'ratings',  # Will scale later
            'log_no_of_ratings': 'log_no_of_ratings',
            'name_length': len(str(product_features['name']).split()),  # Calculate on the fly
            'main_category_encoded': 'main_category_encoded',
            'sub_category_encoded': 'sub_category_encoded'
        }
        
        # Create feature array
        X = []
        for feature in features:
            if feature == 'name_length':
                X.append(feature_mapping[feature])
            elif feature == 'ratings_scaled':
                # Simple scaling between 0-1
                X.append(float(product_features[feature_mapping[feature]]) / 5.0)
            else:
                X.append(float(product_features[feature_mapping[feature]]))
        
        # Reshape for prediction
        X = np.array(X).reshape(1, -1)
        
        # Make prediction
        predicted_price = model.predict(X)[0]
        
        # Apply sanity checks and constraints
        # 1. Get category statistics for reasonable bounds
        similar_products = df[(df['main_category'] == product_features['main_category']) & 
                              (df['sub_category'] == product_features['sub_category'])]
        
        if not similar_products.empty:
            # Calculate statistics
            category_min = similar_products['discount_price'].quantile(0.05)  # 5th percentile
            category_max = similar_products['discount_price'].quantile(0.95)  # 95th percentile
            category_median = similar_products['discount_price'].median()
            
            # Constrain prediction to a reasonable range
            # Don't allow more than 50% above or below the category median
            lower_bound = max(category_min, category_median * 0.5)
            upper_bound = min(category_max, category_median * 1.5)
            
            # Check if competitive with similar products
            competitor_price = float(product_features['discount_price'])
            
            # Base recommendation on competitor's price and category stats
            if predicted_price < lower_bound or predicted_price > upper_bound:
                # Model prediction is outside reasonable bounds
                # Use a blend of competitor price and category median
                predicted_price = (competitor_price * 0.6) + (category_median * 0.4)
                
                # Allow undercutting competitor by 5-15% for market entry
                if predicted_price > competitor_price:
                    discount_factor = np.random.uniform(0.85, 0.95)  # 5-15% discount
                    predicted_price = competitor_price * discount_factor
            
        return predicted_price
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Function to generate historical price data
def generate_historical_prices(product):
    # Get historical price columns
    price_cols = [col for col in product.index if col.startswith('discounted_price_')]
    
    if price_cols:
        # Get non-nan prices
        prices = []
        for col in sorted(price_cols, key=lambda x: int(x.split('_')[-1])):
            if pd.notna(product[col]):
                prices.append(float(product[col]))
        
        # Add current price
        prices.append(float(product['discount_price']))
        
        # Generate weekly dates
        today = datetime.now()
        dates = [(today - timedelta(days=(len(prices)-i-1)*7)).strftime('%Y-%m-%d') for i in range(len(prices))]
        
        return dates, prices
    else:
        # If no historical data, return current price only
        return [datetime.now().strftime('%Y-%m-%d')], [float(product['discount_price'])]

# Function to calculate market statistics for a category
def get_category_stats(df, main_category, sub_category=None):
    try:
        filtered_df = df[df['main_category'] == main_category]
        
        if sub_category:
            filtered_df = filtered_df[filtered_df['sub_category'] == sub_category]
        
        if filtered_df.empty:
            return {}
        
        stats = {
            'avg_price': filtered_df['discount_price'].mean(),
            'min_price': filtered_df['discount_price'].min(),
            'max_price': filtered_df['discount_price'].max(),
            'median_price': filtered_df['discount_price'].median(),
            'avg_discount': ((filtered_df['actual_price'] - filtered_df['discount_price']) / filtered_df['actual_price']).mean() * 100,
            'product_count': len(filtered_df),
            'avg_rating': filtered_df['ratings'].mean(),
        }
        
        return stats
    except Exception as e:
        st.write("No Products found with your search.")

def get_distinct_values(index, field, es):
    response = es.search(index="data_ecom", body={
    "size": 0,
    "aggs": {
        "unique_categories": {
            "terms": {
                "field": f"{field}.keyword",  
                "size": 10000
            }
        }
    }})
    unique_values = [bucket["key"] for bucket in response["aggregations"]["unique_categories"]["buckets"]]
    return unique_values
        
# User search from UI
def search_products(index_name,search_query,sub_category,main_category, es):
    """Search for relevant products based on selected filters."""
    query = {
        "size": 8,  # Limit results to 8
        "query": {
            "bool": {
                "must": []
            }
        }
    }

    # Apply filters dynamically
    # if main_category and main_category != "All":
    #     query["query"]["bool"]["must"].append({"match": {"main_category.keyword": main_category}})
    
    if sub_category:
        query["query"]["bool"]["must"].append({"match": {"sub_category.keyword": sub_category}})
    
    if main_category:
        query["query"]["bool"]["must"].append({"match": {"main_category.keyword": main_category}})
    
    if search_query != "":
        query["query"]["bool"]["must"].append({"match": {"name": {"query": search_query, "fuzziness": "AUTO" }}})
    response = es.search(index=index_name, body=query)
    return response["hits"]["hits"]


# Function to show search page
def show_search_page():
    st.title("🔍 Amazon Seller Price Advisor")
    st.write("Find similar products to help determine optimal pricing for your new Amazon listing")
    es,index_name = es_conn()
    # df = load_data_from_csv()
    

    if es:
        main_categories = get_distinct_values(index_name, "main_category",es)
        sub_categories = get_distinct_values(index_name, "sub_category",es)
        # Add category selection first
        # categories = sorted(main_categories)
        selected_main_category = st.selectbox("Select your product's main category", main_categories)
        
        # Filter subcategories based on main category
        # subcategories = sorted(sub_categories)
        selected_sub_category = st.selectbox("Select your product's subcategory", sub_categories)
        
        # Add search input
        search_query = st.text_input("Search for similar products (by keywords, features, etc.)", "")
        
        # Apply filters
        # filtered_df = df.copy()
        # filtered_df = filtered_df[filtered_df['main_category'] == selected_main_category]
        # filtered_df = filtered_df[filtered_df['sub_category'] == selected_sub_category]
        products_extracted = search_products(index_name, search_query,selected_sub_category,selected_main_category, es)
        
        # Show results count and category statistics
        st.markdown("---")
        st.subheader(f"Market Analysis: {selected_main_category} > {selected_sub_category}")

        products_list=[]
        for index, product in enumerate(products_extracted):
                products_list.append(product["_source"])
        # if search_query:
        #     filtered_df = filtered_df[filtered_df['name'].str.contains(search_query, case=False, na=False)]
        
        
        # Convert the list of dictionaries to a DataFrame
        products = pd.DataFrame(products_list)
        # Get and display category statistics
        cat_stats = get_category_stats(products, selected_main_category, selected_sub_category)
        
        if cat_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Price", f"₹{int(cat_stats['avg_price']):,}")
                st.metric("Products in Category", f"{cat_stats['product_count']}")
            with col2:
                st.metric("Median Price", f"₹{int(cat_stats['median_price']):,}")
                st.metric("Avg. Rating", f"⭐ {cat_stats['avg_rating']:.1f}")
            with col3:
                st.metric("Price Range", f"₹{int(cat_stats['min_price']):,} - ₹{int(cat_stats['max_price']):,}")
            with col4:
                st.metric("Avg. Discount", f"{cat_stats['avg_discount']:.1f}%")
        
        st.markdown("---")
        st.subheader("Similar Products")
        st.write(f"Found {len(products)} comparable products")
        
        if not products.empty:
            # Create three-column layout for products
            num_cols = 3
            for i in range(0, min(len(products), 30), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(products):
                        product = products.iloc[idx]
                        with cols[j]:
                            st.markdown("---")
                            
                            # Display product name (truncated if too long)
                            name_display = str(product['name'])
                            if len(name_display) > 50:
                                name_display = name_display[:50] + "..."
                            st.markdown(f"**{name_display}**")
                            
                            # Display rating
                            if pd.notna(product['ratings']):
                                st.write(f"⭐ {product['ratings']} ({int(product['no_of_ratings']):,} reviews)")
                            
                            # Display price information
                            discount = round(((product['actual_price'] - product['discount_price']) / product['actual_price']) * 100)
                            st.markdown(f"**₹{int(product['discount_price']):,}** <span style='text-decoration: line-through;'>₹{int(product['actual_price']):,}</span> ({discount}% off)", unsafe_allow_html=True)
                            
                            # Button to view competitor analysis
                            if st.button(f"Analyze Competitor", key=f"view_{idx}"):
                                st.session_state.selected_product = product
                                st.session_state.page = 'prediction'
                                st.rerun()
        else:
            st.warning("No comparable products found. Try broadening your search criteria.")
    else:
        st.error("Failed to load data.")

# Function to show prediction page
def show_prediction_page():
    if st.session_state.selected_product is None:
        st.error("No product selected. Please go back and select a comparable product.")
        return

    product = st.session_state.selected_product
    df = data_load()

    # Back button
    if st.button("← Back to Search"):
        st.session_state.page = 'search'
        st.rerun()

    st.title("Competitor & Price Analysis")

    # Product details section
    st.markdown("### Competitor Product Details")
    st.markdown(f"**{product['name']}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Category:** {product['main_category']} > {product['sub_category']}")
        st.write(f"**Rating:** ⭐ {product['ratings']} ({int(product['no_of_ratings']):,} reviews)")
    
    with col2:
        # Price information
        discount = round(((product['actual_price'] - product['discount_price']) / product['actual_price']) * 100)
        st.write(f"**Current Price:** ₹{int(product['discount_price']):,}")
        st.write(f"**List Price:** ₹{int(product['actual_price']):,} ({discount}% off)")
    
    # Market positioning
    st.markdown("---")
    st.markdown("### Market Positioning")
    
    # Get similar products for comparison
    similar_products = df[(df['main_category'] == product['main_category']) & 
                          (df['sub_category'] == product['sub_category'])].copy()
    
    # Create distribution plot of prices in this category
    if not similar_products.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        
        # Distribution plot
        sns.histplot(similar_products['discount_price'], kde=True, ax=ax)
        
        # Add vertical line for current product
        plt.axvline(x=product['discount_price'], color='red', linestyle='--', 
                    label=f'Selected Product (₹{int(product["discount_price"]):,})')
        
        # Calculate recommended price ranges
        low_price = similar_products['discount_price'].quantile(0.25)
        med_price = similar_products['discount_price'].quantile(0.5)
        high_price = similar_products['discount_price'].quantile(0.75)
        
        # Add vertical lines for price segments
        plt.axvline(x=low_price, color='green', alpha=0.5, linestyle=':', 
                    label=f'Budget Range (<₹{int(low_price):,})')
        plt.axvline(x=med_price, color='blue', alpha=0.5, linestyle=':', 
                    label=f'Mid-Range (₹{int(low_price):,}-₹{int(high_price):,})')
        plt.axvline(x=high_price, color='purple', alpha=0.5, linestyle=':', 
                    label=f'Premium Range (>₹{int(high_price):,})')
        
        # Format plot
        plt.xlabel('Price (₹)')
        plt.ylabel('Number of Products')
        plt.title(f'Price Distribution for {product["sub_category"]} Products')
        plt.legend()
        plt.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        # Market segment recommendation
        st.subheader("Market Segment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Budget Price Range", f"<₹{int(low_price):,}")
            if product['discount_price'] < low_price:
                st.info("This competitor targets the budget segment")
        
        with col2:
            st.metric("Mid-Range Price", f"₹{int(low_price):,}-₹{int(high_price):,}")
            if low_price <= product['discount_price'] <= high_price:
                st.info("This competitor targets the mid-range segment")
        
        with col3:
            st.metric("Premium Price Range", f">₹{int(high_price):,}")
            if product['discount_price'] > high_price:
                st.info("This competitor targets the premium segment")
    
    # Prediction section
    st.markdown("---")
    st.markdown("### Recommended Pricing Strategy")
    
    model = load_model()
    
    if model:
        predicted_price = predict_optimal_price(model, product,df)
        
        if predicted_price is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Competitor's Price", f"₹{int(product['discount_price']):,}")
            
            with col2:
                st.metric("Recommended Price", f"₹{int(predicted_price):,}")
            
            with col3:
                price_diff = product['discount_price'] - predicted_price
                price_diff_percent = (price_diff / product['discount_price']) * 100
                
                if price_diff > 0:
                    st.metric("Potential Advantage", f"₹{int(abs(price_diff)):,}", 
                              f"Price {abs(price_diff_percent):.1f}% lower than competitor")
                else:
                    st.metric("Price Gap", f"₹{int(abs(price_diff)):,}", 
                              f"{abs(price_diff_percent):.1f}% higher than competitor")
            
            # Strategy recommendation
            st.markdown("### Pricing Strategy Recommendations")
            
            if price_diff > 0 and price_diff_percent > 15:
                st.success("💰 **Undercut Strategy Available** - You can price significantly lower than this competitor while maintaining profitability.")
                st.write("Consider entering at a lower price point to gain market share quickly.")
            elif price_diff > 0 and price_diff_percent > 5:
                st.info("🏆 **Balanced Advantage** - You can price moderately lower than this competitor.")
                st.write("A slightly lower price can give you an edge while maintaining good margins.")
            elif abs(price_diff_percent) <= 5:
                st.warning("⚖️ **Competitive Match** - This product is priced optimally for the market.")
                st.write("Consider matching this price and differentiating through product features, shipping, or customer service.")
            else:
                st.error("⚠️ **Premium Positioning Needed** - Market indicates a higher price than competitor.")
                st.write("If you choose a higher price, ensure your product offers premium features or quality to justify the cost.")
        else:
            st.warning("Unable to generate price recommendation for this product category.")
    else:
        st.error("Price prediction model not available.")
    
    # Historical price trend
    st.markdown("---")
    st.markdown("### Competitor's Historical Price Trend")
    
    dates, prices = generate_historical_prices(product)
    
    if len(dates) > 1:
        fig, ax = plt.subplots(figsize=(8, 3))
        
        # Plot historical prices
        ax.plot(dates, prices, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        
        # Add predicted price line
        if model and predicted_price is not None:
            ax.axhline(y=predicted_price, color='green', linestyle='--', label='Recommended Price')
        
        # Format the plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.set_title('Competitor Price History')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if model and predicted_price is not None:
            plt.legend()
            
        # Show minimum price point
        min_price = min(prices)
        min_date = dates[prices.index(min_price)]
        ax.annotate(f'₹{int(min_price):,}', 
                    xy=(min_date, min_price),
                    xytext=(10, -20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        st.pyplot(fig)
        
        # Price statistics
        st.markdown("### Price Volatility Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Competitor's Lowest", f"₹{int(min(prices)):,}")
        with col2:
            st.metric("Competitor's Highest", f"₹{int(max(prices)):,}")
        with col3:
            st.metric("Competitor's Average", f"₹{int(sum(prices)/len(prices)):,}")
        with col4:
            price_volatility = (max(prices) - min(prices)) / np.mean(prices) * 100
            st.metric("Price Volatility", f"{price_volatility:.1f}%")
        
        # Market entry strategies
        current_to_min_ratio = product['discount_price'] / min(prices)
        
        if price_volatility > 15:
            st.warning("🔄 **Volatile Market** - This competitor frequently changes prices.")
            st.write("Consider dynamic pricing strategies to remain competitive.")
        
        if current_to_min_ratio <= 1.05:  # Within 5% of historical minimum
            st.error("⚠️ **Aggressive Pricing Period** - Competitor is currently near their historical minimum price.")
            st.write("This may indicate a promotional period or response to competition. Consider timing your entry after this aggressive pricing period ends.")
        elif current_to_min_ratio >= 1.15:  # 15% above historical minimum
            st.success("✅ **Opportune Entry Time** - Competitor is pricing higher than their historical average.")
            st.write("This may be a good time to enter with a competitive price point.")
    else:
        st.info("No historical price data available for this competitor.")

# Main function to route between pages
def main():
    # Sidebar
    with st.sidebar:
        st.title("Amazon Seller Pricing Tool")
        st.markdown("### Features")
        st.markdown("✅ Find Comparable Products")
        st.markdown("✅ Market Analysis")
        st.markdown("✅ Competitor Price Analysis")
        st.markdown("✅ Price Recommendations")
        st.markdown("✅ Entry Strategy Suggestions")
        st.markdown("---")
        
        # Check if model is loaded
        model = load_model()
        # if model:
        #     st.success("✅ Model loaded successfully")
        # else:
        #     st.error("❌ Model not loaded")
    
    # Main content
    if st.session_state.page == 'search':
        show_search_page()
    elif st.session_state.page == 'prediction':
        show_prediction_page()

if __name__ == "__main__":
    collection_name = config['mongodb']['collection_name']
    collection_name_raw = config['mongodb_raw']['collection_name']
    
    # L2_PreprocMongo:  Insert Raw data into collection, Proprocess it and then re-insert to Production Database
    # raw_file_path = "C://Users//jesel sequeira//Downloads//updated_price_tracking_data_raw.csv"
    # insert_file_into_collection_raw(collection_name_raw, raw_file_path)

    # L4_LoadElasticCloud: Load data from Mongodb to ElasticCloud
    # Create indexes - Elastic Search
    # index_mongodb_data(collection_name)

    # LoadUI: Load the Final UI from ElaticCloud
    main()