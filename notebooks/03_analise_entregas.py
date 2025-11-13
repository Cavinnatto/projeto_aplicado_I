import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SHOW_PLOTS = True 

# ————————————————————————————
# Caminhos 
# ————————————————————————————

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
data_path = os.path.join(PROJECT_ROOT, 'data', 'merged_dataset.csv')
images_dir = os.path.join(PROJECT_ROOT, 'images')
outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

print(f"Carregando dados de: {data_path}")

# ————————————————————————————
# Carregar dados
# ————————————————————————————

merged = pd.read_csv(data_path, low_memory=False)

for col in ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']:
    if col in merged.columns:
        merged[col] = pd.to_datetime(merged[col], errors='coerce')

# ————————————————————————————
#  Calcular métricas 
# ————————————————————————————

merged['delivery_time_days'] = (merged['order_delivered_customer_date'] - merged['order_purchase_timestamp']).dt.days
merged['delay_days'] = (merged['order_delivered_customer_date'] - merged['order_estimated_delivery_date']).dt.days
merged['delay_days'] = merged['delay_days'].clip(lower=0) 

# ————————————————————————————
# Estatísticas descritivas
# ————————————————————————————

print("\n📊 Estatísticas de Tempo de Entrega:")
print(merged['delivery_time_days'].describe())
print("\n📊 Estatísticas de Atrasos:")
print(merged['delay_days'].describe())

merged[['delivery_time_days', 'delay_days']].describe().to_csv(
    os.path.join(outputs_dir, 'entregas_descritivas.csv')
)
print(f"Resumo salvo em: {os.path.join(outputs_dir, 'entregas_descritivas.csv')}")

# ————————————————————————————
# Distribuição de tempo de entrega 
# ————————————————————————————

sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))
sns.histplot(merged['delivery_time_days'].dropna(), bins=30, kde=True)
plt.title('Distribuição de Tempo de Entrega (Dias)')
plt.xlabel('Dias')
plt.ylabel('Frequência')
plt.tight_layout()
out_path = os.path.join(images_dir, 'distribuicao_entregas.png')
plt.savefig(out_path, dpi=300)
print(f"Figura salva: {out_path}")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# ————————————————————————————
# Boxplot — Atrasos x Satisfação 
# ————————————————————————————

if 'review_score' in merged.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='review_score', y='delay_days', data=merged)
    plt.title('Atrasos por Review Score')
    plt.xlabel('Review Score')
    plt.ylabel('Dias de Atraso')
    plt.tight_layout()
    out_path = os.path.join(images_dir, 'atrasos_vs_satisfacao.png')
    plt.savefig(out_path, dpi=300)
    print(f"Figura salva: {out_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

# ————————————————————————————
# Tempo médio de entrega por estado 
# ————————————————————————————

if 'customer_state' in merged.columns:
    entregas_por_estado = merged.groupby('customer_state')['delivery_time_days'].mean().sort_values()
    print("\n🚚 Tempo Médio de Entrega por Estado:")
    print(entregas_por_estado.head(10))

    entregas_por_estado.to_csv(os.path.join(outputs_dir, 'tempo_medio_por_estado.csv'))
    print(f"Tabela salva: {os.path.join(outputs_dir, 'tempo_medio_por_estado.csv')}")

    plt.figure(figsize=(10, 8))
    entregas_por_estado.plot(kind='barh', color='skyblue')
    plt.title('Tempo Médio de Entrega por Estado')
    plt.xlabel('Dias Médios')
    plt.ylabel('Estado')
    plt.tight_layout()
    out_path = os.path.join(images_dir, 'tempo_medio_por_estado.png')
    plt.savefig(out_path, dpi=300)
    print(f"Figura salva: {out_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

# ————————————————————————————
# Insight final 
# ————————————————————————————

print("\n Insight: Atrasos maiores correlacionam com scores mais baixos.")
print(" Focar em melhorias logísticas nos estados com maiores tempos médios de entrega.")

print("\n Análise concluída. Resultados em /images e /outputs.")
