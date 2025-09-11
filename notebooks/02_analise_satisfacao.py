import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset merged (gerado no script 01)
merged = pd.read_csv('../data/merged_dataset.csv')

# Análise descritiva da satisfação (review_score)
print("Estatísticas de Satisfação:")
print(merged['review_score'].describe())

# Distribuição de scores
plt.figure(figsize=(8, 6))
sns.histplot(merged['review_score'], bins=5, kde=False)
plt.title('Distribuição de Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Frequência')
plt.savefig('../images/distribuicao_satisfacao.png')  # Salvar gráfico
plt.show()

# Correlação com variáveis numéricas (ex.: preço, frete, tempo de entrega)
merged['delivery_time_days'] = (pd.to_datetime(merged['order_delivered_customer_date']) - 
                                pd.to_datetime(merged['order_purchase_timestamp'])).dt.days
corr_columns = ['review_score', 'price', 'freight_value', 'payment_value', 'delivery_time_days']
corr_matrix = merged[corr_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlações com Review Score')
plt.savefig('../images/correlacoes_satisfacao.png')
plt.show()

# Satisfação por categoria de produto
satisfacao_por_categoria = merged.groupby('product_category_name_english')['review_score'].mean().sort_values(ascending=False)
print("Satisfação Média por Categoria:")
print(satisfacao_por_categoria.head(10))

# Insight: Baixos scores podem estar relacionados a atrasos em entregas ou altos fretes.
