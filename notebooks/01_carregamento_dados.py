import pandas as pd

# Carregar todos os datasets
customers = pd.read_csv('../data/olist_customers_dataset.csv')
order_items = pd.read_csv('../data/olist_order_items_dataset.csv')
order_payments = pd.read_csv('../data/olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('../data/olist_order_reviews_dataset.csv')
orders = pd.read_csv('../data/olist_orders_dataset.csv')
products = pd.read_csv('../data/olist_products_dataset.csv')
sellers = pd.read_csv('../data/olist_sellers_dataset.csv')
category_translation = pd.read_csv('../data/product_category_name_translation.csv')

# Converter colunas de data para datetime
date_columns_orders = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                       'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_columns_orders:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')

date_columns_reviews = ['review_creation_date', 'review_answer_timestamp']
for col in date_columns_reviews:
    order_reviews[col] = pd.to_datetime(order_reviews[col], errors='coerce')

order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'], errors='coerce')

# Merge principal: Orders como base, juntando reviews, items, customers, payments, products e sellers
merged = orders.merge(customers, on='customer_id') \
               .merge(order_reviews, on='order_id', how='left') \
               .merge(order_items, on='order_id', how='left') \
               .merge(order_payments, on='order_id', how='left') \
               .merge(products, on='product_id', how='left') \
               .merge(category_translation, on='product_category_name', how='left') \
               .merge(sellers, on='seller_id', how='left')

# Limpeza básica: Remover duplicatas e nulos críticos
merged = merged.drop_duplicates()
merged = merged.dropna(subset=['order_id', 'customer_id', 'review_score'])  # Foco em satisfação, então drop sem review

# Salvar o dataset merged para uso nos outros scripts
merged.to_csv('../data/merged_dataset.csv', index=False)

print("Dados carregados e merged com sucesso!")
print(merged.info())  # Resumo das colunas
print(merged.head())  # Primeiras linhas
