from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from .models import DatasetCancerBucal
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os
from django.core.files.storage import FileSystemStorage

# Função de login do usuário
def login_usr(request):
    usuario = request.POST.get('username')
    senha = request.POST.get('password')
    user = authenticate(username=usuario, password=senha)
    if user is not None:
        login(request, user)
        request.session['username'] = usuario
        request.session['password'] = senha
        request.session['usernamefull'] = user.get_full_name()
        print(request.session['username'])
        print(request.session['password'])
        print(request.session['usernamefull'])
        return redirect('menu_analitico')
    else:
        data = {}
        if usuario:
            data['msg'] = "Usuário ou senha incorretos!"
        return render(request, 'login_usr.html', data)

# Função do Menu Analítico
def menu_analitico(request):
    return render(request, 'menu_analitico.html')

# Função para importar arquivos de dataset
def ia_import(request):
    return render(request, 'ia_import.html')

# Função de importação do dataset de câncer bucal
def ia_import_save(request):
    if request.method == 'POST' and request.FILES['arq_upload']:
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload']
        file1 = fss.save(upload.name, upload)
        file_url = fss.url(file1)
        DatasetCancerBucal.objects.all().delete()
        
        i = 0
        file2 = open(file1, 'r')
        for row in file2:
            if i > 0:
                row2 = row.replace(',', '.')
                row3 = row2.split(';')
                DatasetCancerBucal.objects.create(
                    grupo=row3[0], tabagismo=float(row3[1]), consumo_alcool=float(row3[2]),
                    idade=float(row3[3]), sexo=float(row3[4]),
                    infeccao_hpv=float(row3[5]), exposicao_solar=float(row3[6]),
                    dieta_inadequada=float(row3[7]), higiene_bucal_inadequada=float(row3[8]),
                    uso_protese_inadequada=float(row3[9]), grau_risco=float(row3[10]))
            i += 1
        file2.close()
    
    return redirect('ia_import_list')

# Função para listar dados importados
def ia_import_list(request):
    data = {}
    data['DatasetCancerBucal'] = DatasetCancerBucal.objects.all()
    return render(request, 'ia_import_list.html', data)

# Função para treinar o modelo KNN
def ia_knn_treino(request):
    data = {}
    print("Modelo em treinamento!")

    # Carregar dados do banco de dados
    dados_queryset = DatasetCancerBucal.objects.all()
    df = pd.DataFrame(list(dados_queryset.values()))
    print("Pandas Carregado e dados 'convertidos'.")
    print("'Cabecalho' dos dados:")
    print(df.head())

    print("Colunas do DataFrame:", df.columns.tolist())

    # Aplicar codificação one-hot na coluna 'grupo'
    #df = pd.get_dummies(df, columns=['grupo'], drop_first=True)

    #df['grupo_Experimental'] = df['grupo_Experimental'].astype(int)

    # Balanceamento de dados (SMOTE)
    X = df.drop(columns=['grupo', 'id'])
    y = df['grupo']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Normalização dos dados
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)

    # Dividir em treino (70%), teste (15%) e validação (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X_res, y_res, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    data['dataset'] = X_train.shape
    data['treino'] = X_train.shape[0]
    data['teste'] = X_test.shape[0]
    data['validacao'] = X_val.shape[0]
    print(f'Tamanho do conjunto de treino: {X_train.shape}')
    print(f'Tamanho do conjunto de teste: {X_test.shape}')
    print(f'Tamanho do conjunto de validação: {X_val.shape}')

    # Ajuste de hiperparâmetros com GridSearchCV
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # Parâmetro para a métrica Minkowski
    }
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Melhor conjunto de parâmetros
    data['best'] = grid_search.best_params_
    print("Melhores parâmetros encontrados:", grid_search.best_params_)

    # Obter o melhor modelo
    best_knn = grid_search.best_estimator_

    # Previsões no conjunto de validação
    y_val_pred = best_knn.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Acurácia no conjunto de validação: {val_accuracy * 100:.2f}%')
    data['acc_validacao'] = round(val_accuracy * 100, 2)

    # Previsões no conjunto de teste
    y_test_pred = best_knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    data['acc_teste'] = round(test_accuracy * 100, 2)
    print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

    # Avaliação de métricas adicionais
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    # Mapear rótulos para valores numéricos
    y_test_numeric = y_test.map({'Controle': 0, 'Experimental': 1})

    # Previsões no conjunto de teste
    y_test_pred = best_knn.predict(X_test)

    # Mapear previsões para valores numéricos
    y_pred_numeric = np.array([0 if pred == 'Controle' else 1 for pred in y_test_pred])

    # AUC-ROC
    y_test_pred_prob = best_knn.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test_numeric, y_test_pred_prob)
    print(f'AUC-ROC: {roc_auc:.2f}')

    # Matriz de Confusão
    cm = confusion_matrix(y_test_numeric, y_pred_numeric)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('static', 'confusion_matrix.png'))
    plt.close()

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test_numeric, y_test_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('static', 'roc_curve.png'))
    plt.close()

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test_numeric, y_test_pred_prob)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join('static', 'precision_recall_curve.png'))
    plt.close()

    # Cross-Validation
    cv_scores = cross_val_score(best_knn, X_res, y_res, cv=5, scoring='roc_auc')
    print(f'Cross-Validation AUC-ROC Scores: {cv_scores}')
    print(f'Mean AUC-ROC: {cv_scores.mean():.2f}')

    # Salvar o modelo treinado
    model_filename = 'knn_model.pkl'
    joblib.dump(best_knn, model_filename)
    print(f'Modelo salvo em: {model_filename}')
    data['file'] = model_filename

    return render(request, 'ia_knn_treino.html', data)

# Função para exibir a matriz de confusão
def ia_knn_matriz(request):
    import joblib
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd
    from .models import DatasetCancerBucal

    dados_queryset = DatasetCancerBucal.objects.all()
    df = pd.DataFrame(list(dados_queryset.values()))
    X = df.drop(columns=['grupo', 'id'])
    y = df['grupo']
    model_filename = 'knn_model.pkl'
    best_knn = joblib.load(model_filename)
    y_pred = best_knn.predict(X)
    cm = confusion_matrix(y, y_pred)
    data = {
        'matrix': cm.tolist(),
        'labels': np.unique(y).tolist()
    }
    for i in data['matrix']:
        print(i)
    return render(request, 'ia_knn_matriz.html', data)

# Função para exibir a curva ROC
def ia_knn_roc(request):
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go
    import numpy as np
    from .models import DatasetCancerBucal

    dados_queryset = DatasetCancerBucal.objects.all()
    df = pd.DataFrame(list(dados_queryset.values()))
    X = df.drop(columns=['grupo', 'id'])
    y = df['grupo'].map({'Controle': -1, 'Experimental': 1})
    model_filename = 'knn_model.pkl'
    best_knn = joblib.load(model_filename)
    y_pred_prob = best_knn.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='RandomClassifier', line=dict(dash='dash', color='gray')))
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falsos Positivos (FPR)',
        yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
        showlegend=True
    )
    graph = fig.to_html(full_html=False)
    return render(request, 'ia_knn_roc.html', {'graph': graph})

# Função para exibir a curva Precision-Recall
def ia_knn_recall(request):
    import joblib
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, auc
    import plotly.graph_objects as go
    import numpy as np
    from .models import DatasetCancerBucal

    dados_queryset = DatasetCancerBucal.objects.all()
    df = pd.DataFrame(list(dados_queryset.values()))
    X = df.drop(columns=['grupo', 'id'])
    y = df['grupo'].map({'Controle': -1, 'Experimental': 1})
    model_filename = 'knn_model.pkl'
    best_knn = joblib.load(model_filename)
    y_pred_prob = best_knn.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_pred_prob)
    pr_auc = auc(recall, precision)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Precision-Recall Curve (AUC = {pr_auc:.2f})', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode='lines', name='RandomClassifier', line=dict(dash='dash', color='gray')))
    fig.update_layout(
        title='Curva Precision-Recall',
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    graph = fig.to_html(full_html=False)
    return render(request, 'ia_knn_recall.html', {'graph': graph})