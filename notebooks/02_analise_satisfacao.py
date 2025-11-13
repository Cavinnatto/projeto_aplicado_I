import os 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set(style='whitegrid')

# Mostrar plots durante a execução?
SHOW_PLOTS = True

# ————————————————————————————
# Caminhos 
# ————————————————————————————
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))  
data_path = os.path.join(PROJECT_ROOT, 'data', 'merged_dataset.csv')
images_dir = os.path.join(PROJECT_ROOT, 'images')
outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')

print(f"BASE_DIR = {BASE_DIR}")
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"Data path esperado: {data_path}")

# ————————————————————————————
# Verificações e criação de pastas 
# ————————————————————————————

if not os.path.exists(data_path):
    print("\nERRO: arquivo de dados não encontrado no caminho acima.")
    print("Verifique se 'merged_dataset.csv' está em:", os.path.join(PROJECT_ROOT, 'data'))
    sys.exit(1)

os.makedirs(images_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# ————————————————————————————
# Carregar dados com parsing de datas 
# ————————————————————————————

print("\nCarregando dados...")
merged = pd.read_csv(data_path, low_memory=False)

# ————————————————————————————
# Converter colunas de data 
# ————————————————————————————

date_cols = ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']
for c in date_cols:
    if c in merged.columns:
        merged[c] = pd.to_datetime(merged[c], errors='coerce')

# ————————————————————————————
#  Limpeza básica 
# ————————————————————————————

merged = merged[~merged['review_score'].isna()].copy()

if 'order_delivered_customer_date' in merged.columns and 'order_purchase_timestamp' in merged.columns:
    merged['delivery_time_days'] = (merged['order_delivered_customer_date'] - merged['order_purchase_timestamp']).dt.days
else:
    merged['delivery_time_days'] = np.nan

if 'order_delivered_customer_date' in merged.columns and 'order_estimated_delivery_date' in merged.columns:
    merged['delay_days'] = (merged['order_delivered_customer_date'] - merged['order_estimated_delivery_date']).dt.days
else:
    merged['delay_days'] = np.nan

merged.loc[merged['delivery_time_days'] < 0, 'delivery_time_days'] = np.nan

# ————————————————————————————
#  Estatísticas descritivas
# ————————————————————————————

desc = merged['review_score'].describe()
counts = merged['review_score'].value_counts().sort_index()
print("\nEstatísticas de review_score:\n", desc)
print("\nContagem por nota:\n", counts)

summary_table = pd.DataFrame({
    'estatistica': desc.index,
    'valor': desc.values
})
summary_table.to_csv(os.path.join(outputs_dir, 'review_score_describe.csv'), index=False)
print(f"\nArquivo salvo: {os.path.join(outputs_dir, 'review_score_describe.csv')}")

# ————————————————————————————
# Histograma discreto (1..5) 
# ————————————————————————————

plt.figure(figsize=(8,6))
bins = [0.5,1.5,2.5,3.5,4.5,5.5]
sns.histplot(merged['review_score'], bins=bins, discrete=True)
plt.title('Distribuição de Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Frequência')
plt.xticks([1,2,3,4,5])
out_path = os.path.join(images_dir, 'distribuicao_satisfacao.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()
print(f"Figura salva: {out_path}")

# ————————————————————————————
# Tratamento de outliers (99º percentil)
# ————————————————————————————

if merged['delivery_time_days'].notna().sum() > 0:
    q99 = merged['delivery_time_days'].quantile(0.99)
    print(f'99º percentil delivery_time_days: {q99}')
    merged['delivery_time_days_trunc'] = merged['delivery_time_days'].clip(upper=q99)
else:
    merged['delivery_time_days_trunc'] = np.nan
    print('Nenhum delivery_time_days válido encontrado.')

# ————————————————————————————
#  Matrizes de correlação (Pearson e Spearman) 
# ————————————————————————————

corr_columns = [c for c in ['review_score', 'price', 'freight_value', 'payment_value', 'delivery_time_days', 'delay_days'] if c in merged.columns]

if len(corr_columns) >= 2:
    pearson = merged[corr_columns].corr(method='pearson')
    spearman = merged[corr_columns].corr(method='spearman')
    pearson.to_csv(os.path.join(outputs_dir, 'corr_pearson.csv'))
    spearman.to_csv(os.path.join(outputs_dir, 'corr_spearman.csv'))
    print(f"Matrizes salvas em {outputs_dir}")


    import numpy as _np
    plt.figure(figsize=(10,8))
    mask = _np.triu(_np.ones_like(pearson, dtype=bool))
    sns.heatmap(pearson, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, vmin=-1, vmax=1)
    out_path = os.path.join(images_dir, 'corr_pearson.png')
    plt.title('Correlação (Pearson) entre variáveis')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print(f"Figura salva: {out_path}")

    plt.figure(figsize=(10,8))
    mask = _np.triu(_np.ones_like(spearman, dtype=bool))
    sns.heatmap(spearman, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, vmin=-1, vmax=1)
    out_path = os.path.join(images_dir, 'corr_spearman.png')
    plt.title('Correlação (Spearman) entre variáveis')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print(f"Figura salva: {out_path}")
else:
    print("Não há colunas suficientes para calcular correlações. Colunas presentes:", corr_columns)

# ————————————————————————————
# Satisfação por categoria 
# ————————————————————————————

if 'product_category_name_english' in merged.columns:
    cat_stats = merged.groupby('product_category_name_english').agg(
        mean_score=('review_score','mean'),
        count=('review_score','count')
    ).sort_values(by='mean_score', ascending=False)
    cat_stats_filtered = cat_stats[cat_stats['count'] >= 30]
    cat_stats_filtered.to_csv(os.path.join(outputs_dir, 'satisfacao_por_categoria.csv'))
    print(f"Satisfação por categoria salva: {os.path.join(outputs_dir, 'satisfacao_por_categoria.csv')}")
else:
    print("Coluna product_category_name_english não encontrada.")

# ————————————————————————————
# Boxplot: delay_days vs review_score
# ————————————————————————————

if 'delay_days' in merged.columns and merged['delay_days'].notna().sum() > 0:
    plt.figure(figsize=(8,6))
    sns.boxplot(x='review_score', y='delay_days', data=merged[~merged['delay_days'].isna()])
    plt.title('Delay (dias) por Review Score')
    plt.xlabel('Review Score')
    plt.ylabel('Delay (dias)')
    out_path = os.path.join(images_dir, 'atrasos_vs_satisfacao.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    print(f"Figura salva: {out_path}")
else:
    print("Não há dados de delay_days suficientes para o boxplot.")

# ————————————————————————————
# Regressão linear robusta 
# ————————————————————————————

model_cols = [c for c in ['delivery_time_days_trunc','delay_days','price','freight_value'] if c in merged.columns]
model_df = merged.dropna(subset=['review_score'] + model_cols).copy()

if len(model_df) > 50 and len(model_cols) > 0:
    X = model_df[model_cols]
    X = sm.add_constant(X)
    y = model_df['review_score']
    ols = sm.OLS(y, X).fit(cov_type='HC3')  
    print(ols.summary())
    coeffs = ols.params.to_frame(name='coef').join(ols.bse.to_frame(name='std_err'))
    coeffs.to_csv(os.path.join(outputs_dir, 'regression_coeffs.csv'))
    print(f"Coeficientes salvos em: {os.path.join(outputs_dir, 'regression_coeffs.csv')}")
else:
    print("Dados insuficientes para regressão (ou variáveis ausentes). model_cols:", model_cols)

print("\nEDA concluída. Verifique as pastas 'images' e 'outputs' no diretório do projeto.")
