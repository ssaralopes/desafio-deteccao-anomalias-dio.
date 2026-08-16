# 🕵️ Detecção de Anomalias em Transações Financeiras

> Desafio de detecção de anomalias em transações financeiras desenvolvido durante minha jornada no **Bootcamp Bradesco - GenAI, Dados & Cyber, da DIO**, utilizando um dataset aberto de transações financeiras como desafio proposto no final do curso **"Análise de Dados com Python: Da Preparação à Aplicação com Segurança"**.

---

## 📌 Sobre o projeto

Este projeto tem como objetivo explorar técnicas de **Análise de Dados e Machine Learning aplicadas à detecção de transações fraudulentas**.

A atividade utiliza um dataset aberto de transações realizadas com cartões de crédito, contendo características numéricas das transações e uma variável que identifica se determinada operação foi considerada fraudulenta ou não.

O projeto foi desenvolvido de forma **experimental e incremental**, começando por uma exploração dos dados e evoluindo gradualmente para diferentes modelos e estratégias de avaliação.

```text
Dataset
   ↓
Exploração dos dados
   ↓
Feature Engineering
   ↓
Logistic Regression
   ↓
Novas Features
   ↓
Logistic Regression novamente
   ↓
Random Forest
   ↓
Avaliação dos modelos
   ↓
Teste de Thresholds
   ↓
Comparação dos resultados
```

Além de buscar um modelo capaz de identificar fraudes, o projeto também foi utilizado como espaço de **estudo e experimentação**, permitindo compreender conceitos de análise de dados, classificação, métricas de avaliação, engenharia de atributos e interpretação de modelos.

---

## 🎯 Objetivos

Os principais objetivos deste desafio foram:

* carregar e explorar um dataset de transações financeiras;
* compreender a distribuição entre transações normais e fraudulentas;
* identificar o problema de **desbalanceamento das classes**;
* realizar uma etapa inicial de Feature Engineering;
* criar novas variáveis para tentar representar padrões das transações;
* treinar um modelo de Logistic Regression como baseline;
* comparar o resultado após a criação de novas features;
* experimentar um modelo de **Random Forest**;
* analisar Precision, Recall e F1-score;
* utilizar Matriz de Confusão;
* avaliar os modelos utilizando **ROC-AUC** e **PR-AUC**;
* analisar a importância das variáveis;
* experimentar diferentes valores de threshold;
* comparar os resultados dos experimentos;
* **documentar os aprendizados e dúvidas encontrados durante o desenvolvimento.**

---

## 📊 Dataset

O dataset utilizado é o **Credit Card Fraud Detection Dataset**, disponibilizado publicamente e contendo transações realizadas por cartões de crédito.

O conjunto possui:

* **284.807 transações**
* **31 colunas**
* **284.315 transações normais**
* **492 transações fraudulentas**

A variável `Class` representa o resultado da transação:

| Valor | Significado           |
| ----- | --------------------- |
| `0`   | Transação normal      |
| `1`   | Transação fraudulenta |

### Distribuição das classes

| Classe       | Quantidade | Percentual |
| ------------ | ---------: | ---------: |
| Normal (`0`) |    284.315 |   99,8273% |
| Fraude (`1`) |        492 |    0,1727% |

Isso significa que as fraudes representam uma parcela muito pequena do conjunto de dados.

Esse cenário é conhecido como **classificação desbalanceada (*imbalanced classification*)**.

---

## 🔐 Variáveis V1 até V28

As colunas `V1` até `V28` são variáveis numéricas que passaram por uma transformação utilizando **PCA (Principal Component Analysis)** no dataset original.

Por questões de privacidade, as características originais das transações não são disponibilizadas de forma identificável.

Por isso, não sabemos exatamente o que `V1`, `V2`, `V3` etc. representam no mundo real.

Mesmo assim, essas variáveis preservam padrões estatísticos que podem ser utilizados pelos modelos para diferenciar transações normais de transações fraudulentas.

Também estão presentes:

* `Time` → tempo decorrido desde o início da coleta;
* `Amount` → valor da transação;
* `Class` → classe da transação, indicando fraude ou não fraude.

---

## 🧰 Tecnologias e ferramentas

O projeto foi desenvolvido utilizando:

* Python
* Pandas
* NumPy
* Scikit-learn
* Logistic Regression
* Random Forest
* Git
* GitHub

---

# 🔎 1. Exploração inicial dos dados

Antes de treinar qualquer modelo, foi realizada uma exploração inicial do dataset.

Foram analisados:

* primeiras linhas da tabela;
* quantidade de registros por classe;
* proporção entre as classes;
* tipos das variáveis;
* estatísticas descritivas;
* valores ausentes.

### Quantidade de classes

```python
df["Class"].value_counts()
```

Resultado:

```text
0    284315
1       492
```

A análise mostrou imediatamente que existe uma diferença muito grande entre as duas classes.

### ⚠️ Por que o desbalanceamento é importante?

Imagine um modelo que simplesmente respondesse:

> "Todas as transações são normais."

Ele acertaria aproximadamente **99,83% das transações**.

À primeira vista, isso pareceria um modelo excelente.

Mas ele **não encontraria nenhuma fraude**.

Esse é um dos motivos pelos quais a análise de fraude não pode depender apenas da métrica de **Accuracy (acurácia)**.

Nesse contexto, métricas como:

* Precision;
* Recall;
* F1-score;
* PR-AUC;

passam a ser especialmente importantes.

---

# 🧠 2. Uma dúvida importante: afinal, o que é uma Feature?

Essa foi uma das dúvidas que surgiu durante o desenvolvimento.

A palavra **feature** pode parecer complicada, mas a ideia é relativamente simples:

> **Feature é uma característica ou informação que o modelo pode utilizar para tentar fazer uma previsão. 🔮**

Por exemplo, imagine uma tabela de transações:

|    Valor | Horário | País   | Fraude |
| -------: | ------- | ------ | -----: |
|    R$ 50 | 10h     | Brasil |      0 |
| R$ 2.500 | 03h     | Brasil |      1 |

Nesse exemplo:

* `Valor` é uma feature;
* `Horário` é uma feature;
* `País` é uma feature;
* `Fraude` é o que queremos prever, portanto é o **target**.

No nosso dataset:

```text
Time
V1
V2
V3
...
V28
Amount
```

são informações que podem ser utilizadas como features.

Já:

```text
Class
```

é o **target**, porque representa aquilo que queremos prever.

---

# 🛠️ 3. E o que é Feature Engineering?

Se uma feature é uma característica utilizada pelo modelo, **Feature Engineering** é o processo de criar, transformar ou combinar informações existentes para tentar fornecer ao modelo características mais úteis.

Uma forma simples de pensar:

```text
Dados originais
      ↓
Transformações
      ↓
Novas informações
      ↓
Features mais úteis
      ↓
Modelo
```

Neste projeto foram testadas algumas features simples, justamente para entender esse processo antes de avançar para técnicas mais complexas.

---

# 🧪 4. Feature Engineering inicial

Na primeira etapa, foi criada uma transformação para a variável `Amount`.

## 💰 Amount_log

Foi criada a variável:

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

Essa transformação aplica uma escala logarítmica ao valor da transação.

Ela pode ser útil quando uma variável possui valores muito dispersos ou alguns valores extremamente altos.

O `log1p()` corresponde a:

```text
log(1 + valor)
```

O `+1` permite trabalhar também com valores iguais a zero.

---

## 📏 Amount_scaled

Também foi criada uma versão padronizada do valor:

```python
scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(
    df[["Amount"]]
)
```

O `StandardScaler` transforma os valores para uma escala baseada em:

```text
média ≈ 0
desvio padrão ≈ 1
```

A ideia é evitar que uma variável com valores numericamente muito grandes tenha uma influência desproporcional em determinados modelos.

> **⚠️ Observação:** nesta primeira versão do estudo, o `StandardScaler` foi aplicado antes da separação entre treino e teste. Em uma implementação mais rigorosa, essa etapa deve ser ajustada para evitar *data leakage*, por exemplo utilizando `Pipeline` e ajustando o scaler somente com os dados de treinamento.

Essa limitação foi mantida na documentação porque faz parte do próprio processo de aprendizagem e representa um ponto importante para uma versão futura do projeto.

---

# 🤖 5. Primeiro experimento — Logistic Regression

O primeiro modelo utilizado foi a **Logistic Regression**.

Apesar do nome conter "Regression", neste contexto ela está sendo utilizada como um algoritmo de **classificação binária**.

O objetivo é classificar cada transação como:

```text
0 → normal
1 → fraude
```

Esse primeiro modelo funciona como um **baseline**.

## O que é um baseline?

Baseline é um ponto de referência.

Primeiro criamos um modelo inicial:

```text
Dados
 ↓
Logistic Regression
 ↓
Resultado inicial
```

Depois podemos alterar alguma coisa e verificar:

> "A mudança realmente melhorou o resultado?"

Sem um baseline, fica mais difícil saber se uma nova técnica realmente trouxe benefício.

---

## ✂️ Separação entre treino e teste

Os dados foram divididos utilizando:

```python
train_test_split()
```

A divisão utilizada foi:

```text
70% → treinamento
30% → teste
```

### Treinamento

O conjunto de treinamento é utilizado para que o modelo aprenda padrões presentes nos dados.

### Teste

O conjunto de teste é separado durante o treinamento e utilizado posteriormente para verificar como o modelo se comporta diante de dados que não foram utilizados para ensiná-lo.

---

## ⚖️ O que significa `stratify=Y`?

Como as classes são muito desbalanceadas, foi utilizado:

```python
stratify=Y
```

Isso ajuda a manter aproximadamente a mesma proporção entre transações normais e fraudulentas nos conjuntos de treinamento e teste.

Assim, evitamos criar uma divisão em que um dos conjuntos tenha uma distribuição muito diferente da original.

---

## 🎲 E o que significa `random_state=42`?

Essa foi outra dúvida que surgiu durante o desenvolvimento.

O processo de divisão dos dados possui aleatoriedade.

O:

```python
random_state=42
```

serve para tornar essa divisão **reproduzível**.

O número `42` não possui um significado especial para o algoritmo.

Poderia ser outro número.

O importante é utilizar o mesmo valor quando queremos reproduzir o experimento.

---

# 📈 6. Avaliação do primeiro modelo

O modelo baseline apresentou:

| Métrica | Resultado |
| ------- | --------: |
| ROC-AUC |    0,9469 |
| PR-AUC  |    0,7039 |

A matriz de confusão foi:

```text
[[85278    17]
 [   54    94]]
```

Podemos interpretar:

|                  | Previsto: Normal | Previsto: Fraude |
| ---------------- | ---------------: | ---------------: |
| **Real: Normal** |           85.278 |               17 |
| **Real: Fraude** |               54 |               94 |

Isso significa que:

* **85.278** transações normais foram identificadas corretamente;
* **17** transações normais foram classificadas como fraude;
* **94** fraudes foram identificadas corretamente;
* **54** fraudes não foram identificadas pelo modelo.

---

# 🧪 7. Segundo experimento — Novas Features

Depois do baseline, foram criadas novas variáveis para verificar se informações adicionais poderiam ajudar o modelo.

Foram criadas:

* `Amount_high`
* `Time_hour`
* `V_mean`
* `V_std`

---

## 💰 Amount_high

A variável identifica transações que estão entre os **5% maiores valores** do dataset.

```python
amount_threshold = df["Amount"].quantile(0.95)

df["Amount_high"] = (
    df["Amount"] > amount_threshold
).astype(int)
```

Resultado:

```text
0 → não está entre os 5% maiores valores
1 → está entre os 5% maiores valores
```

A ideia foi testar se transações de valor excepcionalmente alto poderiam apresentar alguma relação com fraude.

---

## ⏰ Time_hour

A variável `Time` representa o tempo decorrido desde o início da coleta.

Foi criada uma nova variável agrupando esse tempo em horas:

```python
df["Time_hour"] = (
    df["Time"] // 3600
).astype(int)
```

A ideia foi testar se o momento em que uma transação ocorre poderia apresentar algum padrão relevante.

---

## 📊 V_mean

As variáveis `V1` até `V28` foram utilizadas para criar uma média:

```python
df["V_mean"] = df[v_cols].mean(axis=1)
```

Essa variável representa uma medida agregada dos valores das variáveis transformadas para cada transação.

---

## 📉 V_std

Também foi calculado o desvio padrão das variáveis `V1` até `V28`:

```python
df["V_std"] = df[v_cols].std(axis=1)
```

Essa variável representa o quanto esses valores variam entre si em cada transação.

---

# 🤖 8. Logistic Regression com novas Features

Após a criação das novas features, a Logistic Regression foi treinada novamente.

A ideia foi manter o mesmo algoritmo utilizado no baseline e modificar principalmente as informações fornecidas ao modelo.

Dessa forma, a comparação fica mais clara:

### Experimento 1

```text
Features originais
       ↓
Logistic Regression
       ↓
Resultado
```

### Experimento 2

```text
Features originais
       +
Novas Features
       ↓
Logistic Regression
       ↓
Resultado
```

### Resultado

| Métrica | Baseline | Novas Features |
| ------- | -------: | -------------: |
| ROC-AUC |   0,9469 |         0,9504 |
| PR-AUC  |   0,7039 |         0,7098 |

As novas features produziram uma **pequena melhora nos dois indicadores**.

Isso não significa que qualquer feature nova necessariamente melhora um modelo.

O resultado mostra justamente a importância de **testar hipóteses e comparar os resultados**.

---

# 🌲 9. Terceiro experimento — Random Forest

No terceiro experimento, foi utilizado o algoritmo **Random Forest**.

Diferentemente do experimento anterior, agora a principal mudança foi o algoritmo utilizado.

As novas features criadas anteriormente foram mantidas.

A ideia passou a ser:

```text
Mesmas features
      ↓
Random Forest
      ↓
Comparação com Logistic Regression
```

---

## 🌳 O que é Random Forest?

Random Forest é um algoritmo baseado em várias árvores de decisão.

Em vez de depender de uma única árvore, o algoritmo combina várias árvores para chegar a uma decisão.

Uma forma simplificada de imaginar:

```text
             Dados
               │
       ┌───────┼───────┐
       ↓       ↓       ↓
    Árvore  Árvore  Árvore
       │       │       │
       └───────┼───────┘
               ↓
        Decisão combinada
```

Neste projeto, foram utilizadas **100 árvores**:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

---

# 📊 10. Resultado do Random Forest

O modelo apresentou:

| Métrica | Resultado |
| ------- | --------: |
| ROC-AUC |    0,9239 |
| PR-AUC  |    0,8239 |

A matriz de confusão foi:

```text
[[85290     5]
 [   34   114]]
```

Podemos interpretá-la como:

|                  | Previsto: Normal | Previsto: Fraude |
| ---------------- | ---------------: | ---------------: |
| **Real: Normal** |           85.290 |                5 |
| **Real: Fraude** |               34 |              114 |

Portanto:

* **85.290** transações normais foram identificadas corretamente;
* **5** transações normais foram classificadas como fraude;
* **114** fraudes foram identificadas corretamente;
* **34** fraudes não foram identificadas.

Para a classe de fraude, o modelo apresentou:

```text
Precision = 0,96
Recall    = 0,77
F1-score  = 0,85
```

---

# 🧠 11. Entendendo Precision, Recall e F1-score

Durante o desenvolvimento, uma das partes que mais exigiu atenção foi entender exatamente o que cada métrica significava.

---

## 🎯 Precision

Precision responde:

> **"Quando o modelo disse que era fraude, ele estava certo?"**

No Random Forest:

```text
Precision = 0,96
```

Isso significa que, entre as transações classificadas pelo modelo como fraude, aproximadamente **96% realmente eram fraudulentas**.

Os demais casos são **falsos positivos**.

---

## 🔎 Recall

Recall responde:

> **"Das fraudes que realmente existiam, quantas o modelo conseguiu encontrar?"**

No Random Forest:

```text
Recall = 0,77
```

Existiam:

```text
148 fraudes
```

O modelo encontrou:

```text
114
```

e deixou passar:

```text
34
```

Essas 34 transações são os **falsos negativos**.

---

## ⚖️ F1-score

O F1-score combina Precision e Recall em uma única métrica.

A fórmula é:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

No Random Forest:

```text
Precision = 0,96
Recall    = 0,77

F1 ≈ 0,85
```

O F1-score tenta representar o equilíbrio entre:

```text
"Quando acuso fraude, estou certo?"
                    +
"Estou conseguindo encontrar as fraudes?"
```

---

# 👮 12. A analogia do "fiscal" do F1-score

Uma das analogias que mais me ajudou a entender essas métricas foi imaginar que existe uma equipe fiscalizando o modelo.

### 🕵️ Precision

É o fiscal que pergunta:

> "Quando você acusou alguém de fraude, você estava certo?"

### 🔎 Recall

É o fiscal que pergunta:

> "Você conseguiu encontrar as fraudes que realmente existiam?"

### 👮 F1-score

E o F1 é como um **supervisor que olha para os dois fiscais ao mesmo tempo**.

Ele não pergunta apenas:

> "Você acertou?"

Nem apenas:

> "Você encontrou?"

Ele quer saber se o modelo está conseguindo **equilibrar os dois objetivos**.

Por isso:

```text
Precision → acertei quando acusei?
Recall    → encontrei as fraudes?
F1        → consegui equilibrar os dois?
```

Essa analogia foi especialmente útil para entender por que o F1-score **não significa "quantas vezes o modelo disse fraude e realmente era fraude"**.

Essa definição corresponde à **Precision**.

---

# 📈 13. ROC-AUC

Outra métrica utilizada no projeto foi a **ROC-AUC**.

A ROC-AUC avalia a capacidade do modelo de separar as duas classes considerando diferentes pontos de decisão (*thresholds*).

De forma simplificada:

> Quanto maior a ROC-AUC, melhor o modelo consegue distinguir as classes ao variar o threshold.

Os resultados foram:

| Modelo                         | ROC-AUC |
| ------------------------------ | ------: |
| Logistic Regression — Baseline |  0,9469 |
| Logistic Regression + Features |  0,9504 |
| Random Forest                  |  0,9239 |

Neste experimento, a Logistic Regression apresentou ROC-AUC superior.

Porém, como o dataset é extremamente desbalanceado, não é interessante analisar somente essa métrica.

---

# 🎯 14. PR-AUC

A **PR-AUC** representa a área sob a curva Precision-Recall.

Ela é especialmente interessante em problemas onde uma classe é muito menor que a outra, como acontece neste projeto.

Aqui:

```text
99,827% → transações normais
0,173%  → fraudes
```

Os resultados foram:

| Modelo                         | PR-AUC |
| ------------------------------ | -----: |
| Logistic Regression — Baseline | 0,7039 |
| Logistic Regression + Features | 0,7098 |
| Random Forest                  | 0,8239 |

Nesse caso, o **Random Forest apresentou um resultado significativamente melhor**.

Isso mostra por que é importante analisar mais de uma métrica.

---

# 📊 15. Comparação dos experimentos

Os três experimentos foram:

1. Logistic Regression como baseline;
2. Logistic Regression com novas features;
3. Random Forest com as novas features.

## Resultado geral

| Modelo                               | ROC-AUC | PR-AUC |
| ------------------------------------ | ------: | -----: |
| Logistic Regression — Baseline       |  0,9469 | 0,7039 |
| Logistic Regression — Novas Features |  0,9504 | 0,7098 |
| Random Forest — Novas Features       |  0,9239 | 0,8239 |

## Interpretação

A criação de novas features produziu uma pequena melhora na Logistic Regression:

```text
ROC-AUC
0,9469 → 0,9504
```

e:

```text
PR-AUC
0,7039 → 0,7098
```

Já o Random Forest apresentou:

```text
ROC-AUC = 0,9239
PR-AUC  = 0,8239
```

Apesar de apresentar uma ROC-AUC menor, o Random Forest apresentou o **melhor PR-AUC**, que é uma métrica particularmente relevante neste problema de classificação altamente desbalanceada.

Portanto, considerando o objetivo de analisar a identificação da classe minoritária, o Random Forest apresentou um resultado interessante neste experimento.

---

# 🔍 16. Importância das variáveis

Uma das vantagens do Random Forest é permitir analisar uma estimativa da importância das variáveis utilizadas pelo modelo.

As dez variáveis que apresentaram maior importância foram:

| Variável | Importância |
| -------- | ----------: |
| V17      |    0,158019 |
| V14      |    0,139392 |
| V12      |    0,120849 |
| V10      |    0,062668 |
| V11      |    0,057154 |
| V16      |    0,049879 |
| V18      |    0,042099 |
| V_mean   |    0,032464 |
| V9       |    0,030296 |
| V4       |    0,026561 |

As variáveis `V17`, `V14` e `V12` apresentaram as maiores importâncias no modelo.

Como as variáveis `V1` até `V28` foram transformadas por PCA e não possuem significado original disponível, não é possível afirmar diretamente que determinada variável representa, por exemplo, "horário", "localização" ou "tipo de compra".

O que podemos afirmar é que essas variáveis apresentaram **maior importância para as decisões do Random Forest dentro deste conjunto de dados**.

> **Importante:** a importância de variável do Random Forest não significa causalidade. Ela indica a contribuição relativa da variável para as decisões do modelo, dentro da metodologia utilizada.

---

# 🎚️ 17. Teste de Thresholds

Outro experimento realizado foi a alteração do **threshold** utilizado para classificar uma transação como fraude.

O threshold representa o ponto a partir do qual uma probabilidade é convertida em uma classe.

Por exemplo:

```text
Probabilidade de fraude = 0,80

Threshold 0,50 → fraude
Threshold 0,70 → fraude
Threshold 0,90 → não fraude
```

Portanto, mudar o threshold altera o comportamento do modelo.

Foram testados:

```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
```

---

## 📉 Threshold 0,3

```text
Precision = 0,85
Recall    = 0,82
F1-score  = 0,84
```

O modelo encontrou uma parcela maior das fraudes.

Porém, houve aumento de falsos positivos.

---

## ⚖️ Threshold 0,5

```text
Precision = 0,94
Recall    = 0,77
F1-score  = 0,85
```

Esse é o threshold utilizado como referência padrão na classificação binária.

---

## 📈 Threshold 0,7

```text
Precision = 0,98
Recall    = 0,68
F1-score  = 0,80
```

Nesse caso, o modelo ficou mais rigoroso antes de classificar uma transação como fraude.

Isso aumentou a Precision, mas reduziu o Recall.

---

# ⚖️ 18. O trade-off entre Precision e Recall

O teste de thresholds mostrou na prática que existe um equilíbrio entre essas métricas.

De forma simplificada:

```text
Threshold menor
      ↓
Mais transações podem ser classificadas como fraude
      ↓
Recall tende a aumentar
      ↓
Mais falsos positivos podem aparecer
```

Enquanto:

```text
Threshold maior
      ↓
O modelo fica mais rigoroso
      ↓
Precision tende a aumentar
      ↓
Algumas fraudes podem passar despercebidas
      ↓
Recall tende a diminuir
```

Em um sistema financeiro real, a escolha do threshold dependeria do custo de cada tipo de erro.

Por exemplo:

* deixar uma fraude passar pode gerar prejuízo financeiro;
* bloquear uma transação legítima pode prejudicar um cliente.

Portanto, não existe necessariamente um threshold universalmente "melhor".

---

# 🔐 19. Relação com Segurança da Informação

Embora o projeto tenha sido desenvolvido principalmente com técnicas de **Análise de Dados e Machine Learning**, o problema estudado possui uma relação direta com **Segurança da Informação**.

A detecção de transações fraudulentas pode fazer parte de uma estratégia maior de prevenção, detecção e resposta a incidentes.

Em um ambiente real, um sistema poderia funcionar de forma semelhante a:

```text
Transação
    ↓
Análise automática
    ↓
Modelo de detecção
    ↓
┌──────────────────────┐
│ Transação normal     │ → Permitir
└──────────────────────┘

OU

┌──────────────────────┐
│ Transação suspeita   │
└──────────────────────┘
          ↓
       Alerta
          ↓
     Investigação
          ↓
   Decisão / Resposta
```

Esse tipo de abordagem pode se conectar a processos de:

* monitoramento;
* detecção de ameaças;
* análise de comportamento;
* investigação;
* resposta a incidentes;
* prevenção de perdas.

O projeto, portanto, representa uma ponte entre **Dados, Inteligência Artificial e Cybersecurity**.

---

# 🧠 20. O que eu aprendi com este desafio

Este projeto começou como um exercício de detecção de anomalias, mas acabou envolvendo conceitos muito maiores do que simplesmente treinar um modelo.

Entre os principais aprendizados:

### 📊 Análise de dados

Aprendi a realizar uma exploração inicial do dataset antes de começar a modelagem, observando:

* estrutura;
* tipos de dados;
* estatísticas;
* valores ausentes;
* distribuição das classes.

### 🧩 Features

Entendi que uma **feature é uma característica utilizada pelo modelo para realizar uma previsão**.

Também compreendi que **Feature Engineering** consiste em criar ou transformar características existentes para tentar fornecer informações mais úteis ao modelo.

### 🤖 Machine Learning

Tive contato prático com:

* Logistic Regression;
* Random Forest;
* treinamento e teste;
* classificação binária;
* probabilidades de previsão.

### 📈 Avaliação

Aprendi que simplesmente olhar para a acurácia pode ser enganoso em datasets desbalanceados.

Passei a analisar:

* Precision;
* Recall;
* F1-score;
* Matriz de Confusão;
* ROC-AUC;
* PR-AUC.

### 🎚️ Threshold

Entendi que a previsão de um modelo não precisa ser tratada apenas como "0 ou 1".

O modelo pode produzir uma probabilidade e o threshold define a partir de qual ponto essa probabilidade será convertida em uma classe.

### 🔬 Experimentação

Uma das principais lições foi entender que Machine Learning envolve experimentação.

Em vez de simplesmente perguntar:

> "Qual modelo é melhor?"

é necessário perguntar:

> **"Melhor para qual objetivo e considerando qual métrica?"**

---

# ❓ 21. Dúvidas que surgiram durante o desenvolvimento

Este repositório também registra algumas dúvidas que apareceram durante a construção do projeto.

A ideia é mostrar que o processo de aprendizagem não foi apenas escrever código, mas entender o significado por trás dele.

### "O que é uma feature?"

Uma feature é uma característica ou informação que o modelo utiliza para tentar identificar padrões e realizar uma previsão.

### "O F1 significa False 1?"

Não.

O nome **F1-score** não significa "False 1".

Ele é uma métrica que combina **Precision e Recall**, buscando representar o equilíbrio entre as duas.

### "Então F1 é a porcentagem de vezes que o modelo disse fraude e realmente era fraude?"

Também não.

Essa definição corresponde à **Precision**.

```text
Precision → quando disse fraude, estava certo?
Recall    → encontrou as fraudes que realmente existiam?
F1        → equilibrou Precision e Recall?
```

### "Então o F1 é como um fiscal?"

Sim. Essa foi a analogia utilizada durante o estudo:

> **Precision e Recall avaliam o modelo por critérios diferentes, e o F1 funciona como um supervisor que considera os dois resultados para chegar a uma avaliação equilibrada.**

Essa comparação ajudou a transformar uma fórmula abstrata em uma ideia mais intuitiva.

---

# 🚧 22. Limitações do projeto

Este projeto possui caráter **educacional e experimental** e não representa um sistema de detecção de fraude pronto para produção.

Algumas limitações importantes são:

* utilização de um dataset público;
* variáveis `V1` a `V28` sem interpretação direta;
* ausência de informações contextuais reais dos clientes;
* ausência de dados comportamentais completos;
* ausência de custos financeiros associados aos diferentes tipos de erro;
* ausência de uma estratégia completa de otimização para produção;
* aplicação inicial do `StandardScaler` antes da separação entre treino e teste;
* ausência de técnicas específicas de tratamento do forte desbalanceamento das classes.

Em um sistema real, também seriam necessários mecanismos de:

* monitoramento contínuo;
* atualização dos modelos;
* detecção de *data drift* e *concept drift*;
* análise comportamental;
* explicabilidade;
* controle de falsos positivos;
* segurança dos dados;
* privacidade;
* conformidade regulatória.

---

# 🔮 23. Melhorias futuras

Este projeto representa uma primeira implementação e pode ser evoluído de diversas formas.

Algumas possibilidades para futuras versões são:

* testar técnicas de undersampling;
* testar técnicas de oversampling;
* utilizar SMOTE;
* experimentar XGBoost;
* realizar ajuste de hiperparâmetros;
* utilizar `Pipeline` do Scikit-learn;
* corrigir a etapa de padronização para evitar data leakage;
* testar diferentes conjuntos de features;
* comparar diferentes thresholds de maneira sistemática;
* criar visualizações das métricas;
* utilizar SHAP para interpretação mais aprofundada do modelo;
* testar estratégias específicas para dados altamente desbalanceados;
* avaliar o impacto de diferentes métricas conforme o objetivo do sistema;
* criar uma API para disponibilizar o modelo;
* desenvolver um dashboard de monitoramento;
* simular um fluxo de alerta de fraude.

Essas melhorias representam possíveis próximos passos tanto para a evolução técnica do projeto quanto para o aprofundamento dos estudos.

---

# 📁 24. Estrutura do projeto

```text
📁 deteccao-anomalias-transacoes
│
├── 📄 README.md
│
├── 🐍 01_logistic_regression_baseline.py
│
├── 🐍 02_feature_engineering.py
│
└── 🐍 03_random_forest.py
```

### `01_logistic_regression_baseline.py`

Contém:

* coleta dos dados;
* exploração inicial;
* análise das classes;
* Feature Engineering inicial;
* divisão entre treino e teste;
* Logistic Regression;
* Matriz de Confusão;
* Classification Report;
* ROC-AUC;
* PR-AUC.

### `02_feature_engineering.py`

Contém:

* coleta e exploração dos dados;
* criação de novas features;
* `Amount_high`;
* `Time_hour`;
* `V_mean`;
* `V_std`;
* Logistic Regression com novas features;
* avaliação do segundo experimento.

### `03_random_forest.py`

Contém:

* preparação dos dados;
* utilização das novas features;
* Random Forest;
* Classification Report;
* Matriz de Confusão;
* ROC-AUC;
* PR-AUC;
* importância das variáveis;
* teste de diferentes thresholds.

---

# ▶️ 25. Como executar

## 1. Clone o repositório

```bash
git clone <https://github.com/ssaralopes/desafio-deteccao-anomalias-dio..git>
```

## 2. Acesse a pasta

```bash
cd desafio-deteccao-anomalias-dio
```

## 3. Instale as dependências

```bash
pip install pandas numpy scikit-learn
```

## 4. Execute os experimentos

### Experimento 1 — Baseline

```bash
python 01_logistic_regression_baseline.py
```

### Experimento 2 — Feature Engineering

```bash
python 02_feature_engineering.py
```

### Experimento 3 — Random Forest

```bash
python 03_random_forest.py
```

O dataset é carregado diretamente pela URL utilizada no código, portanto não é necessário armazenar o arquivo `creditcard.csv` dentro do repositório.

---

# 📚 26. Conceitos estudados

Durante o desenvolvimento deste desafio, foram trabalhados conceitos relacionados a:

```text
Python
   ↓
Pandas / NumPy
   ↓
Análise exploratória
   ↓
Feature Engineering
   ↓
Classificação
   ↓
Logistic Regression
   ↓
Random Forest
   ↓
Matriz de Confusão
   ↓
Precision / Recall / F1
   ↓
ROC-AUC / PR-AUC
   ↓
Threshold
   ↓
Importância das variáveis
   ↓
Detecção de anomalias
   ↓
Segurança da Informação
```

---

# 🎓 27. Contexto acadêmico e de formação

Este projeto foi desenvolvido como atividade final do curso:

> **"Análise de Dados com Python: Da Preparação à Aplicação com Segurança"**

durante o:

> **Bootcamp Bradesco - GenAI, Dados & Cyber — DIO**

O desafio foi utilizado não apenas como entrega da atividade, mas também como oportunidade para aprofundar conceitos que ainda estavam sendo estudados durante o desenvolvimento.

Por isso, este repositório registra tanto a implementação quanto parte do **processo de aprendizagem**.

---

# 💭 28. Reflexão final

Uma das principais conclusões deste projeto foi perceber que construir um modelo de Machine Learning não significa apenas escolher um algoritmo e observar sua acurácia.

É necessário compreender:

```text
Que dados estou utilizando?
          ↓
O que cada variável representa?
          ↓
Como as classes estão distribuídas?
          ↓
Quais características podem ajudar?
          ↓
Qual modelo faz sentido?
          ↓
Como vou avaliar o resultado?
          ↓
Qual tipo de erro é mais importante?
          ↓
O que os resultados realmente significam?
```

No caso da detecção de fraudes, isso se torna ainda mais importante porque uma pequena classe de transações fraudulentas está escondida dentro de uma quantidade muito maior de transações legítimas.

O projeto também mostrou que um modelo não deve ser considerado "melhor" apenas porque possui uma métrica maior.

Nos experimentos realizados:

* a Logistic Regression apresentou o melhor **ROC-AUC**;
* a Logistic Regression com novas features apresentou uma pequena melhora em relação ao baseline;
* o Random Forest apresentou o melhor **PR-AUC**;
* o Random Forest apresentou um equilíbrio interessante entre Precision e Recall;
* a alteração do threshold modificou o comportamento do modelo;
* métricas diferentes podem contar histórias diferentes sobre o mesmo modelo.

Portanto, uma das principais lições foi entender que a escolha de um modelo depende do **problema que estamos tentando resolver e do tipo de erro que estamos dispostos a aceitar**.

Mais do que encontrar um modelo "perfeito", este projeto foi uma oportunidade para aprender a **questionar os dados, testar hipóteses, interpretar métricas e entender as limitações dos resultados**.

E talvez esse tenha sido o principal aprendizado:

> **Machine Learning não é apenas fazer o modelo funcionar. É entender por que ele funciona, onde ele pode falhar e se o resultado realmente faz sentido para o problema que estamos tentando resolver.**

---

## 🌙 Contato

Se este projeto despertou sua curiosidade, você pode me encontrar por aqui:

[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/ssaralopes)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/ssaralopes/)

---
<p align="center">
  📚 <strong>Bootcamp:</strong> Bradesco - GenAI, Dados & Cyber | 🏫 <strong>Plataforma:</strong> DIO | 🐍 <strong>Linguagem:</strong> Python | 🔐 <strong>Tema:</strong> Detecção de anomalias em transações financeiras
</p>
