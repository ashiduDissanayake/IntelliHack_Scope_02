# Customer Segmentation Project

This project implements customer segmentation for an e-commerce platform based on customer behavior data. The goal is to identify distinct customer segments for targeted marketing strategies.


## Dataset Features

- `customer_id`: Unique identifier for the customer
- `total_purchases`: Total number of purchases made by the customer
- `avg_cart_value`: Average value of items in the customer's cart (in monetary units)
- `total_time_spent`: Total time spent on the platform (in minutes)
- `product_click`: Number of products viewed by the customer
- `discount_count`: Number of times the customer used a discount code

## Expected Customer Segments

1. **Bargain Hunters**: Customers who make frequent purchases of low-value items and heavily rely on discounts
2. **High Spenders**: Premium buyers who focus on high-value purchases and are less influenced by discounts
3. **Window Shoppers**: Customers who spend significant time browsing but rarely make purchases

## Installation

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Place the dataset in the `data/` directory

## Usage

### Using the main script

Run the entire pipeline using:

This will:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Find the optimal number of clusters
4. Build and evaluate clustering models
5. Analyze clusters and identify customer segments
6. Export results to the `output/` directory

### Using individual modules

You can also run each component separately:

```python
# Example: Data preprocessing
from src.data_preprocessing import preprocess_data

original_df, preprocessed_df, scaler = preprocess_data("./data/customer_behavior_analytcis.csv")
