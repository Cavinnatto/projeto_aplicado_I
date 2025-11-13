import pandas as pd
import os

# ————————————————————————————
# Caminhos
# ————————————————————————————
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data')

print(f"Diretório do script: {script_dir}")
print(f"Raiz do projeto: {project_root}")
print(f"Pasta de dados: {data_path}\n")

if not os.path.exists(data_path):
    print(f"Pasta 'data' não encontrada. Criando em: {data_path}")
    os.makedirs(data_path, exist_ok=True)

# ————————————————————————————
# Datasets
# ————————————————————————————
print("Carregando datasets...")
customers = pd.read_csv(os.path.join(data_path, 'olist_customers_dataset.csv'))
order_items = pd.read_csv(os.path.join(data_path, 'olist_order_items_dataset.csv'))
order_payments = pd.read_csv(os.path.join(data_path, 'olist_order_payments_dataset.csv'))
order_reviews = pd.read_csv(os.path.join(data_path, 'olist_order_reviews_dataset.csv'))
orders = pd.read_csv(os.path.join(data_path, 'olist_orders_dataset.csv'))
products = pd.read_csv(os.path.join(data_path, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(data_path, 'olist_sellers_dataset.csv'))
category_translation = pd.read_csv(os.path.join(data_path, 'product_category_name_translation.csv'))

print("✓ Todos os datasets carregados!")

# ————————————————————————————
# Converter data para datetime
# ————————————————————————————
print("\nConvertendo datas...")
date_columns_orders = [
    'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
    'order_delivered_customer_date', 'order_estimated_delivery_date'
]
for col in date_columns_orders:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')

date_columns_reviews = ['review_creation_date', 'review_answer_timestamp']
for col in date_columns_reviews:
    order_reviews[col] = pd.to_datetime(order_reviews[col], errors='coerce')

order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'], errors='coerce')

print("✓ Datas convertidas!")

# ————————————————————————————
# Merge principal
# ————————————————————————————
print("\nRealizando merge dos datasets...")
merged = (
    orders.merge(customers, on='customer_id')
          .merge(order_reviews, on='order_id', how='left')
          .merge(order_items, on='order_id', how='left')
          .merge(order_payments, on='order_id', how='left')
          .merge(products, on='product_id', how='left')
          .merge(category_translation, on='product_category_name', how='left')
          .merge(sellers, on='seller_id', how='left')
)
print(f"✓ Merge concluído! Shape: {merged.shape}")

# ————————————————————————————
# Limpeza
# ————————————————————————————
print("\nRealizando limpeza dos dados...")
registros_antes = len(merged)
merged = merged.drop_duplicates()
registros_apos_duplicatas = len(merged)
print(f"  - Removidas {registros_antes - registros_apos_duplicatas} linhas duplicadas")

merged = merged.dropna(subset=['order_id', 'customer_id', 'review_score'])
registros_final = len(merged)
print(f"  - Removidas {registros_apos_duplicatas - registros_final} linhas sem review_score")
print(f"  - Total de registros finais: {registros_final}")

merged = merged.loc[:, ~merged.columns.duplicated()].copy()

# ————————————————————————————
# Salvar dataset merged
# ————————————————————————————
output_path = os.path.join(data_path, 'merged_dataset.csv')
merged.to_csv(output_path, index=False)
print(f"\n✓ Dataset salvo em: {output_path}")

# Estatísticas básicas
print("\n" + "="*50)
print("RESUMO DO DATASET MERGED")
print("="*50)
print(merged.info())
print("\n" + "="*50)
print("PRIMEIRAS LINHAS")
print("="*50)
print(merged.head())

print("\n" + "="*50)
print("ESTATÍSTICAS BÁSICAS")
print("="*50)
print(f"Total de pedidos: {merged['order_id'].nunique()}")
print(f"Total de clientes: {merged['customer_id'].nunique()}")
print(f"Total de produtos: {merged['product_id'].nunique()}")
print(f"Total de vendedores: {merged['seller_id'].nunique()}")
print(f"Período: {merged['order_purchase_timestamp'].min()} até {merged['order_purchase_timestamp'].max()}")
print(f"Review score médio: {merged['review_score'].mean():.2f}")

