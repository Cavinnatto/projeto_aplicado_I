import pandas as pd
import matplotlib.pyplot as plt

reviews = pd.read_csv("data/olist_dataset/olist_order_reviews_dataset.csv")

# Distribuição das notas de satisfação
reviews['review_score'].value_counts().sort_index().plot(kind='bar')
plt.title("Distribuição das notas de satisfação (1 a 5)")
plt.xlabel("Nota")
plt.ylabel("Quantidade de avaliações")
plt.show()

# Percentual de clientes satisfeitos
satisfeitos = (reviews['review_score'] >= 4).mean() * 100
print(f"Clientes satisfeitos: {satisfeitos:.2f}%")
