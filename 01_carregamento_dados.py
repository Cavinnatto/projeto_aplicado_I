import pandas as pd

# Carregar datasets principais
orders = pd.read_csv("data/olist_dataset/olist_orders_dataset.csv")
reviews = pd.read_csv("data/olist_dataset/olist_order_reviews_dataset.csv")
payments = pd.read_csv("data/olist_dataset/olist_order_payments_dataset.csv")

# Exibir primeiras linhas
print(orders.head())
print(reviews.head())
print(payments.head())

# Estatísticas iniciais
print(reviews['review_score'].describe())
