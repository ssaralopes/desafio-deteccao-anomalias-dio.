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

# URL do dataset utilizado no projeto
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# O pandas lê o arquivo CSV diretamente pela URL
df = pd.read_csv(url)

# Visualiza as primeiras linhas da tabela
print(df.head())


# ============================================================
# 2. CLASSES
# ============================================================

print("\n===== Classes =====")

# Mostra quantas transações existem em cada classe
# 0 = transação normal
# 1 = transação fraudulenta
print(df["Class"].value_counts())

# Mostra a proporção de cada classe
print(df["Class"].value_counts(normalize=True))

# Multiplica a proporção por 100 para visualizar em porcentagem
print(df["Class"].value_counts(normalize=True) * 100)


# ============================================================
# 3. EXPLORAÇÃO INICIAL DOS DADOS
# ============================================================

print("\n===== Infos do DataSet =====")

# Mostra informações sobre as colunas, tipos de dados
# e quantidade de valores não nulos
print(df.info())


print("\n===== Estatisticas descritivas =====")

# Mostra estatísticas como média, desvio padrão,
# valores mínimos, máximos e quartis
print(df.describe())


print("\n===== Valores ausentes =====")

# Soma todos os valores ausentes do DataFrame
# O resultado esperado neste dataset é 0
print(df.isnull().sum().sum())


# ============================================================
# 4. FEATURE ENGINEERING INICIAL
# ============================================================

# Criação de uma nova coluna a partir de Amount.
#
# log1p aplica uma transformação logarítmica que ajuda
# a reduzir o efeito de valores muito altos.
#
# O +1 permite trabalhar também com valores iguais a zero.
df["Amount_log"] = np.log1p(df["Amount"])


# Padronização da variável Amount.
#
# O StandardScaler transforma os valores para uma escala
# baseada em:
# média = 0
# desvio padrão = 1
#
# Isso é útil porque algumas variáveis podem ter escalas
# muito diferentes entre si.

scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(
    df[["Amount"]]
)


# ============================================================
# 5. SEPARAÇÃO ENTRE VARIÁVEIS E CLASSE
# ============================================================

# X contém as variáveis que o modelo poderá utilizar
# para fazer suas previsões.
X = df.drop("Class", axis=1)

# Y contém aquilo que queremos prever:
# 0 = normal
# 1 = fraude
Y = df["Class"]


# ============================================================
# 6. DIVISÃO ENTRE TREINO E TESTE
# ============================================================

# Dividimos os dados em:
# 70% para treinamento
# 30% para teste
#
# O conjunto de treinamento é usado para ensinar o modelo.
# O conjunto de teste é usado posteriormente para verificar
# se o modelo consegue fazer previsões em dados que não utilizou
# durante o treinamento.

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    stratify=Y,        # mantém aproximadamente a mesma proporção
                       # de fraude e não fraude nos dois conjuntos

    test_size=0.3,     # 30% dos dados serão utilizados para teste

    random_state=42    # fixa a aleatoriedade para que a divisão
                       # seja reproduzível em novas execuções
)


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

# Modelo de classificação.
#
# Neste projeto, o modelo tenta prever duas categorias:
#
# 0 = transação normal
# 1 = transação fraudulenta

model = LogisticRegression(max_iter=5000)

# O modelo aprende os padrões utilizando os dados de treinamento
model.fit(X_train, Y_train)


# Faz as previsões de classe:
# 0 ou 1
Y_pred = model.predict(X_test)

# Retorna a probabilidade de cada transação pertencer
# à classe 1 (fraude).
Y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 8. AVALIAÇÃO DO MODELO
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        Y_test,
        Y_pred
    )
)


# ------------------------------------------------------------
# Matriz de Confusão
# ------------------------------------------------------------

print("\n===== Matriz de Confusao =====")

cm = confusion_matrix(
    Y_test,
    Y_pred
)

print(cm)


# ------------------------------------------------------------
# ROC-AUC
# ------------------------------------------------------------

print("\n===== ROC-AUC =====")

# Mede a capacidade do modelo de separar as duas classes
# considerando diferentes thresholds.
roc_auc = roc_auc_score(
    Y_test,
    Y_prob
)

print(roc_auc)


# ------------------------------------------------------------
# PR-AUC
# ------------------------------------------------------------

print("\n===== PR-AUC =====")

# Avalia a relação entre Precision e Recall,
# sendo especialmente útil quando existe grande
# desequilíbrio entre as classes.
pr_auc = average_precision_score(
    Y_test,
    Y_prob
)

print(pr_auc)
