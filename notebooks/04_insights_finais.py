import pandas as pd

payments = pd.read_csv("data/olist_dataset/olist_order_payments_dataset.csv")
reviews = pd.read_csv("data/olist_dataset/olist_order_reviews_dataset.csv")

merged = payments.merge(reviews, on='order_id')

# Média de satisfação por forma de pagamento
print(merged.groupby('payment_type')['review_score'].mean())

# Faixa de valor e satisfação
merged['faixa_valor'] = pd.qcut(merged['payment_value'], 4, labels=["baixo", "médio", "alto", "muito alto"])
print(merged.groupby('faixa_valor')['review_score'].mean())
