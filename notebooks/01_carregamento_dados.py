import pandas as pd

orders = pd.read_csv("data/olist_dataset/olist_orders_dataset.csv")
reviews = pd.read_csv("data/olist_dataset/olist_order_reviews_dataset.csv")
payments = pd.read_csv("data/olist_dataset/olist_order_payments_dataset.csv")

print(orders.head())
print(reviews.head())
print(payments.head())

print(reviews['review_score'].describe())
