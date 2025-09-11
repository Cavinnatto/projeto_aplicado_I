import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset merged
merged = pd.read_csv('../data/merged_dataset.csv')

# Calcular tempo de entrega e atraso
merged['delivery_time_days'] = (pd.to_datetime(merged['order_delivered_customer_date']) - 
                                pd.to_datetime(merged['order_purchase_timestamp'])).dt.days
merged['delay_days'] = (pd.to_datetime(merged['order_delivered_customer_date']) - 
                        pd.to_datetime(merged['order_estimated_delivery_date'])).dt.days
merged['delay_days'] = merged['delay_days'].clip(lower=0)  # Apenas atrasos positivos

# Estatísticas de entregas
print("Estatísticas de Tempo de Entrega:")
print(merged['delivery_time_days'].describe())
print("\nEstatísticas de Atrasos:")
print(merged['delay_days'].describe())

# Distribuição de tempo de entrega
plt.figure(figsize=(8, 6))
sns.histplot(merged['delivery_time_days'].dropna(), bins=30, kde=True)
plt.title('Distribuição de Tempo de Entrega (Dias)')
plt.xlabel('Dias')
plt.ylabel('Frequência')
plt.savefig('../images/distribuicao_entregas.png')
plt.show()

# Impacto de atrasos na satisfação
plt.figure(figsize=(10, 6))
sns.boxplot(x=merged['review_score'], y=merged['delay_days'])
plt.title('Atrasos por Review Score')
plt.xlabel('Review Score')
plt.ylabel('Dias de Atraso')
plt.savefig('../images/atrasos_vs_satisfacao.png')
plt.show()

# Entregas por estado do cliente
entregas_por_estado = merged.groupby('customer_state')['delivery_time_days'].mean().sort_values()
print("Tempo Médio de Entrega por Estado:")
print(entregas_por_estado)

# Insight: Atrasos maiores correlacionam com scores baixos; focar em logística em estados com tempos altos.
