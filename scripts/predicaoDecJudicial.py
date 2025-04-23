#%% packages install
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install pingouin
!pip install statstests
!pip install fuzzywuzzy[speedup]
!pip install openpyxl
!pip install joblib
 
 
# %% import
import pandas as pd
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statstests.process import stepwise
from statsmodels.iolib.summary2 import summary_col
from statsmodels.discrete.discrete_model import MNLogit
from scipy.interpolate import UnivariateSpline
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import defaultdict  
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score,roc_curve, auc

import random
import string
from joblib import dump
from joblib import load
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

#%% Funçoes
def remover_acentos_trocando_espacos(texto, mudar_por):
    if isinstance(texto, str):
        # Normalize the unicode string
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
        # Replace spaces with the specified character
        texto = texto.replace(' ', mudar_por)
        # Remove special characters
        texto = re.sub(r'[^\w\s]', '', texto)
        return texto
    return texto

#pré-processamento simples (convertendo tudo para minúsculas, removendo espaços e pontuações)
def preprocessar_nome(nome):
    return ''.join(e for e in nome.lower() if e.isalnum())


def agrupar_nomes_diffs_com_id(df, column_ide, column_name, threshold=85):
    df['ID_GRUPO'] = None  # Inicializa a coluna ID_GRUPO como None
    lista_nomes = df[[column_ide, column_name]].drop_duplicates().values.tolist()  # Remove duplicatas e converte para lista

    # Processa os nomes para comparação
    preprocessed_nomes = {id_: preprocessar_nome(nome) for id_, nome in lista_nomes}

    grupo_id = 1  # Inicia um contador de grupos
    ids_agrupados = set()  # Para armazenar IDs já utilizados
    grupos = {}  # Para armazenar grupos

    for i in range(len(lista_nomes)):
        id_base, nome_base = lista_nomes[i]

        if id_base in ids_agrupados:
            continue

        # Inicializa um grupo com o nome base
        grupo = [nome_base]
        ids_agrupados.add(id_base)

        for j in range(i + 1, len(lista_nomes)):
            id_comparar, nome_comparar = lista_nomes[j]
            nome_base_preprocessado = preprocessed_nomes[id_base]
            nome_comparar_preprocessado = preprocessed_nomes[id_comparar]

            # Verifica similaridade e se os nomes são diferentes
            if fuzz.ratio(nome_base_preprocessado, nome_comparar_preprocessado) >= threshold and nome_base != nome_comparar:
                grupo.append(nome_comparar)
                ids_agrupados.add(id_comparar)

        if len(grupo) > 1:  # Apenas adiciona grupos com mais de um nome
            grupos[grupo_id] = grupo
            # Atribui o ID do primeiro nome do grupo ao ID_GRUPO de todos os nomes no grupo
            for nome in grupo:
                df.loc[df[column_name] == nome, 'ID_GRUPO'] = id_base  # Usa o ID do primeiro nome
            grupo_id += 1

    return df  # Retorna o DataFrame atualizado


def remover_acentos_trocando_espacos(texto, mudar_por):
    if isinstance(texto, str):
        return unicodedata.normalize('NFKD', texto.replace(' ', mudar_por)).encode('ASCII','ignore').decode('utf-8')
    return texto


def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

def preencher_regiao(df, uf_column='UF', regiao_column='REGIAO'):
    #constante
    uf_para_regiao = {
        'AC': 'Norte', 'AL': 'Nordeste', 'AM': 'Norte', 'AP': 'Norte', 'BA': 'Nordeste',
        'CE': 'Nordeste', 'DF': 'Centro-Oeste', 'ES': 'Sudeste', 'GO': 'Centro-Oeste', 
        'MA': 'Nordeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MG': 'Sudeste',
        'PA': 'Norte', 'PB': 'Nordeste', 'PR': 'Sul', 'PE': 'Nordeste', 'PI': 'Nordeste', 
        'RJ': 'Sudeste', 'RN': 'Nordeste', 'RS': 'Sul', 'RO': 'Norte', 'RR': 'Norte', 
        'SC': 'Sul', 'SP': 'Sudeste', 'SE': 'Nordeste', 'TO': 'Norte'
    }
    
    # Criar a nova coluna 'REGIAO' baseada no dicionário de mapeamento
    df[regiao_column] = df[uf_column].map(uf_para_regiao)
    return df

name_mapping = {}

def process_data(data):
    # Verifica se o input é um DataFrame ou um dicionário
    is_dataframe = isinstance(data, pd.DataFrame)
    
    # Se for um DataFrame, faça uma cópia para evitar modificações diretas
    if is_dataframe:
        df = data.copy()
    else:
        # Se for um dicionário, converta para DataFrame
        df = pd.DataFrame([data])

    # Remover colunas não necessárias
    if is_dataframe:
        df = df.drop(['IDENTIFICADOR', 'ACAO', 'QUARTER_ACAO', 'LEGAL_REGIME', 'PEDIDOS'], axis=1)

    # Processar JUIZ
    df[['JUIZ_TRATADO', 'DAC_IDE_JUIZ']] = df['JUIZ'].str.split(' {} ', expand=True)
    df['DAC_IDE_JUIZ'] = df['DAC_IDE_JUIZ'].replace(['', None], 0).fillna(0).astype('int64')

    # Processar ESCRITORIO_ADVERSO
    df[['ESCRITORIO_ADVERSO_TRATADO', 'DAC_IDE_EA']] = df['ESCRITORIO_ADVERSO'].str.split(' {} ', expand=True)
    df['DAC_IDE_EA'] = df['DAC_IDE_EA'].replace(['', None], 0).fillna(0).astype('int64')

    # Preencher valores ausentes em PEDIDO
    df['PEDIDO'] = df['PEDIDO'].fillna('NAO_INFORMADO')

    # Aplicar transformações de texto
    text_columns = ['COMARCA', 'PEDIDO', 'APLICATIVO', 'KIND', 'EQUIPE']
    for col in text_columns:
        df[col + '_TRATADO'] = df[col].apply(lambda x: remover_acentos_trocando_espacos(x, '_')).str.upper()

    # Processar a coluna SUCESSO
    if 'SUCESSO' in df.columns:
        df.loc[df['SUCESSO'] == 'sim', 'SUCESSO'] = 1
        df.loc[df['SUCESSO'] != 1, 'SUCESSO'] = 0
        df['SUCESSO'] = df['SUCESSO'].astype('int64')

    # Retornar o DataFrame processado ou o dicionário original com as transformações aplicadas
    return df if is_dataframe else df.iloc[0].to_dict()

def ajustar_processo_predicao(data):
    # Verifica se o input é um DataFrame ou um dicionário
    is_dataframe = isinstance(data, pd.DataFrame)
    
    # Se for um DataFrame, faça uma cópia para evitar modificações diretas
    if is_dataframe:
        df = data.copy()
    else:
        # Se for um dicionário, converta para DataFrame
        df = pd.DataFrame([data])

    #separar campos IDs e Nomenclaturas de Dado Cliente Juiz
    df[['JUIZ_TRATADO', 'IDE_JUIZ']] = df['JUIZ'].str.split(' {} ', expand=True)
    df['IDE_JUIZ'] = df['IDE_JUIZ'].replace(['', None], 0).fillna(0).astype('int64')

    #separar campos IDs e Nomenclaturas de Dado Cliente ESCRITORIO_ADVERSO
    df[['ESCRITORIO_DEFESA', 'IDE_ED']] = df['ESCRITORIO_DEFESA'].str.split(' {} ', expand=True)
    df['IDE_ED'] = df['IDE_ED'].replace(['', None], 0).fillna(0).astype('int64')
    
    # Preencher valores ausentes em PEDIDO
    df['PEDIDO_ORIGINAL'] = df['PEDIDO_ORIGINAL'].fillna('NAO_INFORMADO')
    
    # criar novos campos aplicando as padronizações de texto nas colunas da lista text_columns
    text_columns = ['COMARCA', 'PEDIDO_ORIGINAL', 'APLICATIVO', 'KIND', 'EQUIPE']
    for col in text_columns:
        df[col + '_TRATADO'] = df[col].apply(lambda x: remover_acentos_trocando_espacos(x, '_')).str.upper()


def plot_conditional_density(data, column_name, hue_name, title):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x=column_name, hue=hue_name, fill=True, alpha=0.3, palette='viridis')
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel('Densidade')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#%% Carregando dados modelo e mantendo apenas linhas e colunas pertinentes
df_decisoes_ori = pd.read_excel('data/dados.xlsx')
df_decisoes_ori.info()

#%% [PRIMEIROS PASSOS - tratando nao utilizados e missing values]
df_decisoes2 = df_decisoes_ori.drop(['IDENTIFICADOR'], axis = 1)
df_decisoes2.info()

df_decisoes2 = df_decisoes2[df_decisoes_ori['PEDIDO_ORIGINAL'].notnull()]

#df_decisoes2[df_decisoes2['JUIZ_TRATADO'] == 'NAO_IDENTIFICADO']['JUIZ_TRATADO'].count()
df_decisoes2 = df_decisoes2[df_decisoes2['JUIZ'] != 'NAO_IDENTIFICADO']

df_decisoes2.info()
#%% [PRIMEIROS PASSOS - 2 e 3] - normalizando textos #####################    Organizando dataframes (1)
df_decisoes2['COMARCA_TRATADO'] = df_decisoes2['COMARCA'].apply(lambda x: remover_acentos_trocando_espacos(x, '_')).str.upper()
df_decisoes2['PEDIDO_ORIGINAL_TRATADO'] = df_decisoes2['PEDIDO_ORIGINAL'].apply(lambda x: remover_acentos_trocando_espacos(x, '_')).str.upper()
df_decisoes2['PRODUTO_TRATADO'] = df_decisoes2['PRODUTO'].apply(lambda x: remover_acentos_trocando_espacos(x, '_')).str.upper()
df_decisoes2['TIPO_TRATADO'] = df_decisoes2['TIPO'].apply(lambda x: remover_acentos_trocando_espacos(x, '_')).str.upper()
df_decisoes2.loc[df_decisoes2['SUCESSO'] =='sim', 'SUCESSO'] = 1
df_decisoes2.loc[df_decisoes2['SUCESSO'] != 1, 'SUCESSO'] = 0
df_decisoes2['SUCESSO'] = df_decisoes2['SUCESSO'].astype('int64')
# ------------------------------------------------------------------------------------------

#%% [AGRUPAMENTO POR SIMILARIDADE] #####################    Ajustando nomes Juiz Promotoria
df_juiz_nomes = df_decisoes2[['JUIZ', 'IDE_JUIZ']]
df_juiz_nomes['JUIZ_TRATADO_COLUNA'] = df_juiz_nomes['JUIZ'].apply(lambda x: remover_acentos_trocando_espacos(x, '_')).str.upper()

# df para agrupamento de nomes semelhantes
df_juiz_nomes_semelhantes = agrupar_nomes_diffs_com_id(df_juiz_nomes, 'IDE_JUIZ', 'JUIZ_TRATADO_COLUNA')

#Mapeando nomes e adicionando a coluna com o valor tratado
mapeamento_nomes = {}
for _, grupo in df_juiz_nomes_semelhantes.iterrows():
    nome_padrao = grupo['JUIZ_TRATADO_COLUNA']  # Escolhe o nome padrão do grupo
    # Adiciona o nome padrão ao mapeamento para cada nome do grupo
    for nome in df_juiz_nomes['JUIZ_TRATADO_COLUNA']:
        # Se os nomes forem semelhantes, atualiza o mapeamento
        if fuzz.ratio(preprocessar_nome(nome), preprocessar_nome(nome_padrao)) >= 93:
            mapeamento_nomes[nome] = nome_padrao           
            
                        
df_juiz_nomes['NOME_TRATADO'] = df_juiz_nomes['JUIZ_TRATADO_COLUNA'].replace(mapeamento_nomes)

#Mescla o dataframe para substituir os nomes que foram tratados e agrupados por suas variações baseadas no identificador
df_decisoes2 = pd.merge(df_decisoes2, df_juiz_nomes[['IDE_JUIZ', 'NOME_TRATADO']], left_on='IDE_JUIZ', right_on='IDE_JUIZ', how='left')
df_decisoes2 = df_decisoes2.drop(['JUIZ'], axis = 1)

df_decisoes2.rename(columns={'JUIZ_TRATADO':'JUIZ'}, inplace=True)
df_decisoes2.rename(columns={'NOME_TRATADO':'JUIZ_TRATADO'}, inplace=True)
df_decisoes2.rename(columns={'IDE_JUIZ':'IDE_JUIZ'}, inplace=True)

#%% [EXTRACAO DE ATRIBUTOS] - Adequando campos de data 

# Se as datas estiverem no formato MM/DD/YYYY ou algo similar:
df_decisoes2['DATA_DIST'] = pd.to_datetime(df_decisoes2['DATA_DIST'], format='%m/%d/%Y', errors='coerce')
df_decisoes2['DATA_ENC'] = pd.to_datetime(df_decisoes2['DATA_ENC'], format='%m/%d/%Y', errors='coerce')

df_decisoes2['DURACAO_PROCESSO'] = (df_decisoes2['DATA_ENC'] - df_decisoes2['DATA_DIST']).dt.days

#%% [AGRUPAMENTO E REDUÇÃO DE CATEGORIAS] - cria agrupamento de categorias =======================     PEDIDO_TRATADO
df_decisoes2['PEDIDO_ORIGINAL_TRATADO'].nunique()
df_decisoes2.groupby('PEDIDO_ORIGINAL_TRATADO')['SUCESSO'].mean()
# executar agrupamento_categorias.py


agrupamento_categorias = {
    'HACKED_ACCOUNT': [
        'NA_-_HACKED_ACCOUNT/A01_-_LINK',
        'NA_-_HACKED_ACCOUNT/A01_-_PIN_CODE',
        'NA_-_HACKED_ACCOUNT/A01_-_SIM_SA01P',
        'LOSS_OF_ACCOUNT_CONTROL_HACKED_ACCOUNT',
        'NA_-_HACKED_ACCOUNT/A01_-_INDERTERMINED'
    ],
    'FAKE_ACCOUNT': [
        'FAKE_ACCOUNT_[PAGE/PROFILE]',
        'NA_-_FAKE_ACCOUNT',
        'IMPERSONATION/UNAUTHORIZED_[ACCOUNT]',
        'IMPERSONATION/UNAUTHORIZED_[PROFILE/PAGE]',
        'NA_-_IMPERSONATION/UNAUTHORIZED_[ACCOUNT]'
    ],
    'PRIVACY_VIOLATION': [
        'NA_-_PRIVACY_VIOLATIONS',
        'NA_-_LGPD',
        'NA_-_LGPD/ANPD_INTERVENTION',
        'NI/IG_-_LGPD',
        'PRIVACY_VIOLATION'
    ],
    'OFFENSIVE_CONTENT': [
        'NA_-_OFFENSIVE_CONTENT',
        'OFFENSIVE_CONTENT_[DIFAM,_INJURIA_OU_CALUNIA]'
    ],
    'ACCOUNT_ISSUES': [
        'ACCOUNT_BAN_/_POLICY_VIOLATION',
        'NA_-_ACCOUNT_BAN_/_POLICY_VIOLATION',
        'ACCOUNT_MANAGEMENT_ISSUES_(PAID_ADS)/_REQUEST_FOR_REFUND_(BOLETO_REFUND)_'
    ],
    'OUTAGE_OR_SERVICE': [
        'NA_-_OUTAGE',
        'TEMPORARY_RESTRICTIONS/OTHER',
        'ACCOUNT/CONTENT_REMOVED_BY_CO'
    ],
    'JOINT_LIABILITY': [
        'JOINT_LIABILITY_/FRAUD_',
        'JOINT_LIABILITY_/SPOOF_ADS'
    ],
    'OTHER_FRAUD': [
        'NA_-_OTHER_FRAUD_ALEGATIONS'
    ]
}

# Inverter o dicionário para fazer o mapeamento correto no DataFrame
agrupamento_invertido = {v: k for k, values in agrupamento_categorias.items() for v in values}

# Aplicar o agrupamento ao DataFrame
df_decisoes2['PEDIDO_ORIGINAL_TRATADO'] = df_decisoes2['PEDIDO_ORIGINAL_TRATADO'].replace(agrupamento_invertido)

# Verificar o resultado
print(df_decisoes2['PEDIDO_ORIGINAL_TRATADO'].value_counts())

#%% [SUAVIZAÇÃO 1] - Aplicar support_smooting para JUIZ PROMOTORIA
# Parâmetro de suavização
k = 5

# Média global do sucesso
media_global = df_decisoes2['SUCESSO'].mean()

# Contagem de casos e média de sucesso por juiz
contagem_por_juiz = df_decisoes2.groupby('JUIZ_TRATADO')['SUCESSO'].count()
media_por_juiz = df_decisoes2.groupby('JUIZ_TRATADO')['SUCESSO'].mean()

# Aplicar a suavização
juiz_suavizado = (contagem_por_juiz * media_por_juiz + k * media_global) / (contagem_por_juiz + k)

# Adicionar a coluna de encoding suavizado ao dataframe
df_decisoes2['JUIZ_SUCESSO_SUAVIZADO'] = df_decisoes2['JUIZ_TRATADO'].map(juiz_suavizado)

# Verificar o resultado
print(df_decisoes2[['JUIZ_TRATADO', 'JUIZ_SUCESSO_SUAVIZADO']].head())

#%% [SUAVIZAÇÃO 2] - Aplicar support_smooting para PEDIDO_TRATADO
# Parâmetro de suavização
k = 5

# Média global do sucesso
#media_global = df_decisoes2['SUCESSO'].mean()

# Contagem de casos e média de sucesso por juiz
contagem_reclamacao = df_decisoes2.groupby('PEDIDO_ORIGINAL_TRATADO')['SUCESSO'].count()
media_por_reclamacao = df_decisoes2.groupby('PEDIDO_ORIGINAL_TRATADO')['SUCESSO'].mean()

# Aplicar a suavização
reclamacao_suavizado = (contagem_reclamacao * media_por_reclamacao + k * media_global) / (contagem_reclamacao + k)

# Adicionar a coluna de encoding suavizado ao dataframe
df_decisoes2['PEDIDO_ORIGINAL_TRATADO_SUCESSO_SUAVIZADO'] = df_decisoes2['PEDIDO_ORIGINAL_TRATADO'].map(reclamacao_suavizado)

# Verificar o resultado
print(df_decisoes2[['PEDIDO_ORIGINAL_TRATADO', 'PEDIDO_ORIGINAL_TRATADO_SUCESSO_SUAVIZADO']].head())

#%% [PADRONIZACAO] - Tratando valores currency
# Selecionar apenas colunas valores
df_decisoes2.info()
colunas_currency = ['CAUSA','CONTINGENCIA']
print(colunas_currency)

df_filtrado = df_decisoes2.drop(['IDE_JUIZ','PRODUTO','PEDIDO_ORIGINAL','PEDIDOS','TIPO','TIPO_TRATADO','COMARCA','COMARCA_TRATADO'], axis=1)
df_filtrado_para_ajustes_posteriores = df_decisoes2


# Verificar se há valores nulos nas colunas numéricas
valores_nulos_currency = df_filtrado[colunas_currency].isnull().sum()
print(valores_nulos_currency)

# por ser menos sensível à outliers, Standard Scaling é a melhor escolha.
scaler = StandardScaler()
df_filtrado[colunas_currency] = scaler.fit_transform(df_filtrado[colunas_currency])

# Verificando o resultado
print(df_filtrado[colunas_currency].head())

#%% [AGRUPAMENTO] - passo extra para remover comarca e UF mudando para região
df_filtrado = preencher_regiao(df_filtrado)
df_filtrado['REGIAO'] = df_filtrado['REGIAO'].str.upper()

#%% ==  ENCODING 1 ==  Prepara df para encodar
df_encoded = pd.get_dummies(df_filtrado.drop(['UF','PEDIDO_ORIGINAL_TRATADO','DATA_DIST','DATA_ENC','JUIZ_TRATADO'], axis=1), columns=['REGIAO', 'PRODUTO_TRATADO'], dtype=int, drop_first=True)
df_filtrado['REGIAO'].value_counts()
# Exibir as primeiras linhas do DataFrame codificado
df_encoded.head()

#%% ==  ENCODING 2 ==  Renomeia colunas do dataframe 
remapEncoded = {
"JUIZ_SUCESSO_SUAVIZADO":"JUIZ",
"PEDIDO_ORIGINAL_TRATADO_SUCESSO_SUAVIZADO":"PEDIDO_ORIGINAL",
"PRODUTO_TRATADO_NAFACALI":"PROD_NAFACALI",
"PRODUTO_TRATADO_NILITALIS":"PROD_NILITALIS",
}

df_dummyzado = df_encoded.copy()
df_dummyzado.rename(columns=remapEncoded, inplace=True)

df_dummyzado.info()
#%% ==  Inicia preparação das matrizes para aplicar modelo
# Variável dependente (target)
y = df_dummyzado['SUCESSO']

# Variáveis independentes
X = df_dummyzado.drop(columns=['SUCESSO'])
X = sm.add_constant(X)


#%% Execução LogisticRegression
# Dividir o conjunto de dados em 80% treino e 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Inicializar o modelo de Regressão Logística
modelo = LogisticRegression(max_iter=1000)
# Treinar o modelo
modelo.fit(X_train, y_train)
# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)
probasLogiReg = modelo.predict_proba(X)# Criar uma nova coluna com a probabilidade da classe positiva
#df_encoded['Probabilidade'] = probas[:, 1]

#%% Calculando as previsões individuais do modelo
y_train_pred = modelo.predict(X_train)
y_test_pred = modelo.predict(X_test)

# Calculando a acurácia
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Acurácia no treino: {train_accuracy}")
print(f"Acurácia no teste: {test_accuracy}")

#%% Avaliação de significancia de modelo

# =============================== PARTE 1
# Avaliar a acurácia
print('Acurácia:', accuracy_score(y_test, y_pred))

# Imprimir a matriz de confusão
print('Matriz de Confusão:')
print(confusion_matrix(y_test, y_pred))

# Relatório de classificação
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))


# =============================== PARTE 2
# Previsões para o conjunto de treino
y_train_pred = modelo.predict(X_train)

# Previsões para o conjunto de teste
y_test_pred = modelo.predict(X_test)

#%% Execução LOGIT
# Ajustar o modelo de regressão logística
# Se necessário voltar para remover variáveis sem significancia estatistica
logit_model = sm.Logit(y, X).fit()
logit_model.summary()
probasLogito = logit_model.predict(X)



#%% -- RESULTADOS E DISCUSSAO  ============================================================ ###################### ==================================================
#==============================         [01. Curvas ROC – Treino x Teste]

# Previsões de probabilidade
y_train_prob = modelo.predict_proba(X_train)[:, 1]  # Probabilidade da classe positiva
y_test_prob = modelo.predict_proba(X_test)[:, 1]

# Cálculo do AUC
auc_train = roc_auc_score(y_train, y_train_prob)
auc_test = roc_auc_score(y_test, y_test_prob)

print(f"AUC no Treino: {auc_train:.4f}")
print(f"AUC no Teste: {auc_test:.4f}")

# Curvas ROC
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

# Configuração das cores usando a paleta viridis
colors = plt.cm.viridis(np.linspace(0, 1, 3))  # Três cores: treino, teste e linha aleatória

# Plot das curvas ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, label=f'Treino (AUC = {auc_train:.4f})', color=colors[0], linestyle='-')
plt.plot(fpr_test, tpr_test, label=f'Teste (AUC = {auc_test:.4f})', color=colors[1], linestyle='--')

# Linha aleatória (modelo sem discriminação)
plt.plot([0, 1], [0, 1], linestyle='--', color=colors[2], label='Aleatório (AUC = 0.5000)')

# Configurações do gráfico
plt.title('Curvas ROC - Treino x Teste', fontsize=14)
plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% ==============================         [02. Eficácia do modelo]
# Calcular acurácia
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Acurácia no Treino: {train_accuracy:.4f}")
print(f"Acurácia no Teste: {test_accuracy:.4f}")


#%% ==============================         [ -  Variáveis comprovadas para multicolinearidade]
X_with_const = sm.add_constant(X)

# Calcular o VIF para cada variável
vif = pd.DataFrame()
vif["Variável"] = X_with_const.columns
vif["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

vifObj = vif

# Exibir o VIF para cada variável
print(vif)
## ===============================================================

#%% ==============================         [03. Relevância das Variáveis Figura_03]

# Extração dos coeficientes do modelo
coef = logit_model.params  # Coeficientes estimados
variables = coef.index      # Nomes das variáveis

# Criar um DataFrame com as variáveis e seus coeficientes
coef_df = pd.DataFrame({'Variável': variables, 'Coeficiente': coef.values})

# Ordenar pela magnitude dos coeficientes (valor absoluto)
coef_df['Magnitude'] = coef_df['Coeficiente'].abs()
coef_df = coef_df.sort_values(by='Magnitude', ascending=True)

# Criar o gráfico de barras horizontais com a paleta viridis
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(coef_df['Coeficiente'].rank(pct=True))  # Aplicar viridis com base no ranking percentual dos coeficientes
sns.barplot(x='Coeficiente', y='Variável', data=coef_df, palette=colors, orient='h')
plt.axvline(0, color='black', linestyle='--')  # Linha vertical em x=0 para indicar neutralidade
plt.title('Importância das Variáveis no Modelo de Regressão Logística')
plt.xlabel('Coeficiente')
plt.ylabel('Variável')
plt.tight_layout()
plt.show()

#%% ==============================         [04. Comportamento de coeficiente positivo]
def plot_conditional_density(data, column_name, hue_name, title):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x=column_name, hue=hue_name, fill=True, alpha=0.3, palette='viridis')
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel('Densidade')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Exemplo de uso com df_encoded
try:
    plot_conditional_density(
        data=df_encoded,
        column_name='CONTINGENCIA',
        hue_name='SUCESSO',
        title='Distribuição da Contingência por Sucesso'
    )
except NameError:
    print("Erro: O DataFrame 'df_encoded' não está definido. Certifique-se de carregar os dados corretamente.")

# Exemplo de uso com df_dummyzado
try:
    plot_conditional_density(
        data=df_dummyzado,
        column_name='CONTINGENCIA',
        hue_name='SUCESSO',
        title='Distribuição da Contingência por Sucesso'
    )
except NameError:
    print("Erro: O DataFrame 'df_dummyzado' não está definido. Certifique-se de carregar os dados corretamente.")

#%% ==============================         [05. Comportamento de coeficiente positivo]
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df_dummyzado,
    x='JUIZ',
    hue='SUCESSO',
    fill=True,
    alpha=0.3,
    palette='viridis'
)

# Personalizar o gráfico
plt.title('Densidade da distribuição de por sucesso', fontweight='bold')
plt.xlabel('Julgamentos por Juiz')
plt.ylabel('Densidade')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

