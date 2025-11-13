import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ————————————————————————————
# CONFIGURAÇÃO DE CAMINHOS 
# ————————————————————————————

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
data_path = os.path.join(PROJECT_ROOT, 'data', 'merged_dataset.csv')
outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

print(f"Carregando base de dados de: {data_path}")

# ————————————————————————————
# CARREGAMENTO DOS DADOS 
# ————————————————————————————

merged = pd.read_csv(data_path, low_memory=False)


for col in ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']:
    if col in merged.columns:
        merged[col] = pd.to_datetime(merged[col], errors='coerce')

if 'delivery_time_days' not in merged.columns and \
   {'order_purchase_timestamp','order_delivered_customer_date'}.issubset(merged.columns):
    merged['delivery_time_days'] = (
        merged['order_delivered_customer_date'] - merged['order_purchase_timestamp']
    ).dt.days

if 'delay_days' not in merged.columns and \
   {'order_estimated_delivery_date','order_delivered_customer_date'}.issubset(merged.columns):
    merged['delay_days'] = (
        merged['order_delivered_customer_date'] - merged['order_estimated_delivery_date']
    ).dt.days.clip(lower=0)

# ————————————————————————————
# INSIGHTS DESCRITIVOS 
# ————————————————————————————

media_satisfacao = merged['review_score'].mean()
porcentagem_baixa_satisfacao = (merged['review_score'] < 3).mean() * 100
media_tempo_entrega = merged['delivery_time_days'].mean()
media_atraso = merged['delay_days'].mean()

print("\n📈 Insights Finais:")
print(f"- Satisfação Média: {media_satisfacao:.2f}/5")
print(f"- % de Baixa Satisfação (<3): {porcentagem_baixa_satisfacao:.2f}%")
print(f"- Tempo Médio de Entrega: {media_tempo_entrega:.2f} dias")
print(f"- Atraso Médio: {media_atraso:.2f} dias")

insights_txt = os.path.join(outputs_dir, 'insights_finais.txt')
with open(insights_txt, 'w', encoding='utf-8') as f:
    f.write("INSIGHTS FINAIS\n")
    f.write(f"Satisfação Média: {media_satisfacao:.2f}/5\n")
    f.write(f"Porcentagem de Baixa Satisfação (<3): {porcentagem_baixa_satisfacao:.2f}%\n")
    f.write(f"Tempo Médio de Entrega: {media_tempo_entrega:.2f} dias\n")
    f.write(f"Atraso Médio: {media_atraso:.2f} dias\n")
print(f"Arquivo salvo: {insights_txt}")

# ————————————————————————————
# MODELO DE REGRESSÃO LINEAR
# ————————————————————————————

features = [c for c in ['delivery_time_days', 'price', 'freight_value'] if c in merged.columns]
if len(features) == 0:
    print("\n Nenhuma variável preditora encontrada (price/freight/time).")
else:
    df_model = merged.dropna(subset=['review_score'] + features).copy()

    X = df_model[features]
    y = df_model['review_score']

    if len(df_model) < 50:
        print("\n Base insuficiente para treino (menos de 50 observações).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        print("\n📊 Resultados do Modelo Linear:")
        print(f"- MSE: {mse:.3f}")
        print(f"- Coeficientes: {dict(zip(features, model.coef_))}")

        coeffs = pd.DataFrame({'variavel': features, 'coeficiente': model.coef_})
        coeffs.to_csv(os.path.join(outputs_dir, 'coeficientes_regressao.csv'), index=False)
        print(f"Coeficientes salvos: {os.path.join(outputs_dir, 'coeficientes_regressao.csv')}")

# ————————————————————————————
# INSIGHT FINAL 
# ————————————————————————————

print("\n Insight Final: Reduzir atrasos pode aumentar significativamente a satisfação.")
print(" Próximo passo: aplicar clusterização para segmentar clientes e personalizar estratégias.")
print("\n Execução concluída com sucesso. Resultados em /outputs/.")
