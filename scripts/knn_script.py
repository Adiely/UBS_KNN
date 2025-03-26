import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carregar os dados
df = pd.read_csv('DatasetCancerBucal.csv', sep=';')

# Aplicar codificação one-hot na coluna 'grupo'
df = pd.get_dummies(df, columns=['grupo'], drop_first=True)

# Converter a coluna 'grupo_Experimental' para inteiro
df['grupo_Experimental'] = df['grupo_Experimental'].astype(int)

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df.drop('grau_risco', axis=1)
y = df['grau_risco']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Aplicar o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avaliar o modelo
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nAcurácia:")
print(accuracy_score(y_test, y_pred))

# Ajuste de hiperparâmetros
param_grid = {'n_neighbors': range(1, 20)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Melhor valor de k:", grid_search.best_params_)

# Reavaliar o modelo com o melhor k
best_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
best_knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred_best = best_knn.predict(X_test)

# Avaliar o modelo
print("Matriz de Confusão (Melhor k):")
print(confusion_matrix(y_test, y_pred_best))

print("\nRelatório de Classificação (Melhor k):")
print(classification_report(y_test, y_pred_best))

print("\nAcurácia (Melhor k):")
print(accuracy_score(y_test, y_pred_best))