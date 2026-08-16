import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. COLETA DE DADOS
# ============================================================

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print(df.head())


# ============================================================
# 2. CLASSES
# ============================================================

print("\n===== Classes =====")

# 0 = transação normal
# 1 = transação fraudulenta
print(df["Class"].value_counts())

# Proporção de cada classe
print(df["Class"].value_counts(normalize=True))

# Proporção em porcentagem
print(df["Class"].value_counts(normalize=True) * 100)


# ============================================================
# 3. EXPLORAÇÃO INICIAL DOS DADOS
# ============================================================

print("\n===== Infos do DataSet =====")
print(df.info())


print("\n===== Estatisticas descritivas =====")
print(df.describe())


print("\n===== Valores ausentes =====")
print(df.isnull().sum().sum())


# ============================================================
# 4. FEATURE ENGINEERING INICIAL
# ============================================================

# Transformação logarítmica do valor da transação.
df["Amount_log"] = np.log1p(df["Amount"])


# Padronização do valor da transação.
scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(
    df[["Amount"]]
)


# ============================================================
# 5. NOVAS FEATURES
# ============================================================

print("\n============================================")
print("NOVAS FEATURES")
print("============================================")


# ------------------------------------------------------------
# 5.1 Amount_high
# ------------------------------------------------------------

# Descobrimos qual é o valor correspondente aos 5%
# maiores valores de transação.
amount_threshold = df["Amount"].quantile(0.95)


# Criamos uma variável que indica:
#
# 0 = valor não está entre os 5% maiores
# 1 = valor está entre os 5% maiores
#
# A ideia é verificar se transações de valor muito alto
# possuem alguma relação com fraudes.

df["Amount_high"] = (
    df["Amount"] > amount_threshold
).astype(int)


# ------------------------------------------------------------
# 5.2 Time_hour
# ------------------------------------------------------------

# A coluna Time representa o tempo decorrido desde
# o início do período analisado.
#
# Dividimos por 3600 para transformar segundos em horas.
#
# A ideia é criar uma variável mais simples para verificar
# se o horário da transação pode apresentar algum padrão.

df["Time_hour"] = (
    df["Time"] // 3600
).astype(int)


# ------------------------------------------------------------
# 5.3 V_mean
# ------------------------------------------------------------

# Selecionamos as variáveis V1 até V28.
#
# Essas variáveis foram transformadas por PCA no dataset
# original para preservar informações sensíveis.

v_cols = [f"V{i}" for i in range(1, 29)]


# Calculamos a média das variáveis V1 até V28
# para cada transação.
#
# A ideia é criar uma medida geral do comportamento
# dessas variáveis naquela transação.

df["V_mean"] = df[v_cols].mean(axis=1)


# ------------------------------------------------------------
# 5.4 V_std
# ------------------------------------------------------------

# Calculamos o desvio padrão das variáveis V1 até V28.
#
# Essa variável pode ajudar a representar o quanto
# os valores das variáveis transformadas variam entre si.

df["V_std"] = df[v_cols].std(axis=1)


# ============================================================
# 6. SEPARAÇÃO ENTRE VARIÁVEIS E CLASSE
# ============================================================

# Agora X contém as variáveis originais e também
# as novas features criadas no experimento.
X_features = df.drop("Class", axis=1)

Y_features = df["Class"]


# ============================================================
# 7. DIVISÃO ENTRE TREINO E TESTE
# ============================================================

X_train_features, X_test_features, Y_train_features, Y_test_features = train_test_split(
    X_features,
    Y_features,
    stratify=Y_features,
    test_size=0.3,
    random_state=42
)


# ============================================================
# 8. LOGISTIC REGRESSION COM NOVAS FEATURES
# ============================================================

# Utilizamos novamente Logistic Regression.
#
# A ideia é manter o algoritmo igual ao experimento anterior
# e mudar principalmente as informações fornecidas ao modelo.
#
# Dessa forma podemos comparar:
#
# Baseline
#        X
#        ↓
# Logistic Regression
#
# Novas Features
#        X + novas variáveis
#        ↓
# Logistic Regression

model_features = LogisticRegression(max_iter=5000)

model_features.fit(
    X_train_features,
    Y_train_features
)


# Previsão das classes
Y_pred_features = model_features.predict(
    X_test_features
)


# Probabilidade de pertencer à classe 1 (fraude)
Y_prob_features = model_features.predict_proba(
    X_test_features
)[:, 1]


# ============================================================
# 9. AVALIAÇÃO
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        Y_test_features,
        Y_pred_features
    )
)


print("\n===== MATRIZ DE CONFUSAO =====")

print(
    confusion_matrix(
        Y_test_features,
        Y_pred_features
    )
)


print("\n===== ROC-AUC =====")

roc_auc_features = roc_auc_score(
    Y_test_features,
    Y_prob_features
)

print(roc_auc_features)


print("\n===== PR-AUC =====")

pr_auc_features = average_precision_score(
    Y_test_features,
    Y_prob_features
)

print(pr_auc_features)


# ============================================================
# 10. RESULTADO DO EXPERIMENTO
# ============================================================

print("\n============================================")
print("RESULTADO DO EXPERIMENTO 2")
print("============================================")

print("ROC-AUC:", roc_auc_features)
print("PR-AUC:", pr_auc_features)

print("\nAs novas features foram adicionadas ao modelo.")
print("Os resultados podem ser comparados com o modelo baseline.")
