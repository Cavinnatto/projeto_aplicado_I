import pandas as pd

orders = pd.read_csv("data/olist_dataset/olist_orders_dataset.csv")
reviews = pd.read_csv("data/olist_dataset/olist_order_reviews_dataset.csv")

# Merge
merged = orders.merge(reviews, on='order_id')

# Converter datas
merged['order_purchase_timestamp'] = pd.to_datetime(merged['order_purchase_timestamp'])
merged['order_delivered_customer_date'] = pd.to_datetime(merged['order_delivered_customer_date'])
merged['order_estimated_delivery_date'] = pd.to_datetime(merged['order_estimated_delivery_date'])

# Calcular atraso
merged['atraso_entrega'] = merged['order_delivered_customer_date'] > merged['order_estimated_delivery_date']

# Taxa de atraso por nota de satisfação
print(merged.groupby('review_score')['atraso_entrega'].mean())
