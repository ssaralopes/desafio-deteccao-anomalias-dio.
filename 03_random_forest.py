import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

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

# Proporção das classes
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

# Identifica transações que estão entre os 5%
# maiores valores do dataset.

amount_threshold = df["Amount"].quantile(0.95)

df["Amount_high"] = (
    df["Amount"] > amount_threshold
).astype(int)


# ------------------------------------------------------------
# 5.2 Time_hour
# ------------------------------------------------------------

# Converte o tempo da transação para uma representação
# em horas.

df["Time_hour"] = (
    df["Time"] // 3600
).astype(int)


# ------------------------------------------------------------
# 5.3 V_mean
# ------------------------------------------------------------

# Seleciona as variáveis V1 até V28.

v_cols = [f"V{i}" for i in range(1, 29)]


# Cria a média das variáveis V1 até V28
# para cada transação.

df["V_mean"] = df[v_cols].mean(axis=1)


# ------------------------------------------------------------
# 5.4 V_std
# ------------------------------------------------------------

# Cria o desvio padrão das variáveis V1 até V28
# para cada transação.

df["V_std"] = df[v_cols].std(axis=1)


# ============================================================
# 6. SEPARAÇÃO ENTRE VARIÁVEIS E CLASSE
# ============================================================

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
# 8. RANDOM FOREST
# ============================================================

print("\n============================================")
print("EXPERIMENTO 3 - RANDOM FOREST")
print("============================================")


# Random Forest é um modelo formado por várias árvores
# de decisão.
#
# n_estimators = quantidade de árvores utilizadas.
#
# random_state = mantém o experimento reproduzível.
#
# n_jobs=-1 permite utilizar todos os núcleos disponíveis
# para acelerar o treinamento.

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Treinamento do modelo
rf_model.fit(
    X_train_features,
    Y_train_features
)


# Previsão das classes
Y_pred_rf = rf_model.predict(
    X_test_features
)


# Probabilidade de cada transação pertencer à classe 1
# (fraude).

Y_prob_rf = rf_model.predict_proba(
    X_test_features
)[:, 1]


# ============================================================
# 9. AVALIAÇÃO DO RANDOM FOREST
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        Y_test_features,
        Y_pred_rf
    )
)


print("\n===== MATRIZ DE CONFUSAO =====")

print(
    confusion_matrix(
        Y_test_features,
        Y_pred_rf
    )
)


print("\n===== ROC-AUC =====")

roc_auc_rf = roc_auc_score(
    Y_test_features,
    Y_prob_rf
)

print(roc_auc_rf)


print("\n===== PR-AUC =====")

pr_auc_rf = average_precision_score(
    Y_test_features,
    Y_prob_rf
)

print(pr_auc_rf)


# ============================================================
# 10. IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

print("\n============================================")
print("IMPORTANCIA DAS VARIAVEIS")
print("============================================")


# Random Forest consegue calcular uma estimativa de
# importância para cada variável utilizada pelo modelo.
#
# Quanto maior o valor, maior foi a contribuição daquela
# variável para as decisões das árvores.

importances = pd.Series(
    rf_model.feature_importances_,
    index=X_features.columns
)


# Mostra as 10 variáveis com maior importância.

print(
    importances
    .sort_values(ascending=False)
    .head(10)
)


# ============================================================
# 11. TESTE DE THRESHOLDS
# ============================================================

print("\n============================================")
print("TESTE DE THRESHOLDS")
print("============================================")


# O threshold padrão geralmente utilizado para classificação
# binária é 0.5.
#
# Aqui vamos testar outros valores para observar como isso
# altera Precision, Recall e F1.
#
# Threshold menor:
# o modelo fica mais disposto a classificar uma transação
# como fraude.
#
# Threshold maior:
# o modelo fica mais rigoroso antes de classificar
# uma transação como fraude.

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]


for threshold in thresholds:

    # Se a probabilidade de fraude for maior ou igual
    # ao threshold escolhido, classificamos como 1.
    #
    # Caso contrário, classificamos como 0.

    Y_pred_threshold = (
        Y_prob_rf >= threshold
    ).astype(int)


    print(f"\nThreshold: {threshold}")

    print(
        classification_report(
            Y_test_features,
            Y_pred_threshold,
            zero_division=0
        )
    )
