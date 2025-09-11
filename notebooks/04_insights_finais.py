import pandas as pd


merged = pd.read_csv('../data/merged_dataset.csv')

# Insights principais
media_satisfacao = merged['review_score'].mean()
porcentagem_baixa_satisfacao = (merged['review_score'] < 3).mean() * 100
media_tempo_entrega = merged['delivery_time_days'].mean()
media_atraso = merged['delay_days'].mean()

print("Insights Finais:")
print(f"- Satisfação Média: {media_satisfacao:.2f}/5")
print(f"- Porcentagem de Baixa Satisfação (<3): {porcentagem_baixa_satisfacao:.2f}%")
print(f"- Tempo Médio de Entrega: {media_tempo_entrega:.2f} dias")
print(f"- Atraso Médio: {media_atraso:.2f} dias")

# Proposta Analítica: Usar regressão linear para prever review_score baseado em delivery_time, price, etc.
# Exemplo simples (pode expandir com sklearn na Etapa 2)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features = ['delivery_time_days', 'price', 'freight_value']
X = merged[features].dropna()
y = merged.loc[X.index, 'review_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"MSE do Modelo de Previsão de Satisfação: {mse:.2f}")
print("Coeficientes:", dict(zip(features, model.coef_)))

# Insight Final: Reduzir atrasos pode aumentar satisfação; proposta: Implementar clustering de clientes para campanhas personalizadas.
