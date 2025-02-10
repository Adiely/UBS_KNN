from django.views.generic import ListView
from django.shortcuts import render

# Define a função index

def index(request):
    return render(request, 'index.html')

# FUNÇÃO PARA IMPORTAR OS ARQUIVOS DE DATASET
def ia_import(request):
    return render (request, 'ia_import.html')


# FUNÇÃO DE IMPORTAÇÃO DO DATASET CANCER BUCAL

def ia_import_save(request):
    from .models import DatasetCancerBucal
    import os
    from django.core.files.storage import FileSystemStorage
    if request.method == 'POST' and request.FILES['arq_upload']:
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload'] 
        file1 = fss.save(upload.name, upload)
        file_url = fss.url(file1)
        DatasetCancerBucal.objects.all().delete() 
        
        i = 0

        file2 = open(file1,'r')
        for row in file2:
            if (i > 0):
                row2 = row.replace(',', '.')
                row3 = row2.split(';')
                DatasetCancerBucal.objects.create(
                                grupo = row3[0], idade = float(row3[1]), sexo = float(row3[2]), 
                                tabagismo = float(row3[3]), consumo_alcool = float(row3[4]), 
                                infeccao_hpv = float(row3[5]), exposicao_solar = float(row3[6]),
                                dieta_inadequada = float(row3[7]), higiene_bucal_inadequada = float(row3[8]), 
                                uso_protese_inadequada = float(row3[9]), grau_risco = float(row3[10]))
            i = i + 1
        file2.close()
        #os.remove(file_url.replace("/", ""))
    
    from django.shortcuts import redirect
    return redirect('ia_import_list') 

def ia_import_list(request):
    from.models import DatasetCancerBucal
    data = {}
    data ['DatasetCancerBucal'] = DatasetCancerBucal.objects.all()
    return render(request, 'ia_import_list.html', data)   

# FUNÇÃO DE IMPORTAÇÃO DO DATASET PACIENTES

'''
def ia_import_save(request):
    from .models import Pacientes
    import os
    from django.core.files.storage import FileSystemStorage
    if request.method == 'POST' and request.FILES['arq_upload']:
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload']
        file1 = fss.save(upload.name, upload)
        file_url = fss.url(file1)
        from .models import Pacientes
        Pacientes.objects.all().delete() 
        i = 0
        
        file2 = open(file1,'r')
        for row in file2:
            if (i > 0):
                row2 = row.replace(',', '.')
                row3 = row2.split(';')
                Pacientes.objects.create(
                                grupo = row3[0], Idade = float(row3[1]), Sexo = float(row3[2]), 
                                Cond_Cancer = float(row3[3]), Cond_Diabetes = float(row3[4]), 
                                Cond_Hipertensao = float(row3[5]), Cond_Obesidade = float(row3[6]),
                                Adm_Emergencia = float(row3[7]), Adm_Urgencia = float(row3[8]), 
                                Med_Ibuprofeno = float(row3[9]), Med_Lipitor = float(row3[10]),                               
                                Med_Paracetamol = float(row3[11]), Med_Penicilina = float(row3[12]),
                                Teste_Inconclusivo = float(row3[13]), Teste_Normal = float(row3[14]),
                                Teste_Anormal = float(row3[15]))
        i = i + 1
        file2.close()
        #os.remove(file_url.replace("/", ""))
    from django.shortcuts import redirect
    return redirect('ia_import_list') 
    
def ia_import_list(request):
    from .models import Pacientes
    data = {}
    data['Pacientes'] = Pacientes.objects.all()
    return render(request, 'ia_import_list.html', data) 
'''  

def ia_knn_treino(request):
    data = {}
    print("Modelo em treinamento!")
    import pandas as pd 
    from.models import DatasetCancerBucal
    dados_queryset= DatasetCancerBucal.objects.all()
    print("Registros Selecionados.")
    df= pd.DataFrame(list(dados_queryset.values()))
    print("Pandas Carregado e dados 'convertidos'.")
    print("'Cabecalho' dos dados:")
    print(df.head())
    from sklearn.model_selection import train_test_split
    print("Sklearn carregado")
    
    # Supondo que 'grupo' seja a variável alvo e o restante são as características (features)
    X= df.drop (columns=['grupo', 'id'])  # Variáveis independentes
    y= df['grupo']  # Variável dependente (target)
    
    # Dividir em treino (70%), teste (15%) e validação (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30,
    random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, 
    random_state=42)
    data['dataset'] = X_train.shape
    data['treino'] = X_train.shape[0]
    data['teste'] = X_test.shape[0]
    data['validacao'] = X_val.shape[0]
    print(f'Tamanho do conjunto de treino: {X_train.shape}')
    print(f'Tamanho do conjunto de teste: {X_test.shape}')
    print(f'Tamanho do conjunto de validação: {X_val.shape}')
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    # Instanciando o KNN
    knn= KNeighborsClassifier()
    # Definindo o grid de parâmetros para o KNN
    param_grid= {
        'n_neighbors': [3, 5, 7, 9],  # Exemplos de valores possíveis
        'weights': ['uniform', 'distance'],  # Tipos de pesos
        'metric': ['euclidean', 'manhattan']  # Tipos de distância
        }
    # Usando o GridSearchCVpara encontrar os melhores parâmetros
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, verbose=2,
    n_jobs=-1)
    
    # Treinando o modelo com os dados de treino
    grid_search.fit(X_train, y_train)
    # Melhor conjunto de parâmetros
    data['best'] = grid_search.best_params_
    print("Melhores parâmetros encontrados:", grid_search.best_params_)
    # Obter o melhor modelo
    best_knn = grid_search.best_estimator_
    # Previsões no conjunto de validação
    y_val_pred = best_knn.predict(X_val)
    # Avaliação do modelo (Accuracy)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Acurácia no conjunto de validação: {val_accuracy* 100:.2f}%')
    data['acc_validacao'] = round(val_accuracy* 100, 2)
    # Previsões no conjunto de teste
    y_test_pred = best_knn.predict(X_test)
    # Avaliação do modelo no conjunto de teste
    test_accuracy = accuracy_score (y_test, y_test_pred)
    data['acc_teste'] = round(test_accuracy* 100, 2)
    print(f'Acurácia no conjunto de teste: {test_accuracy* 100:.2f}%')

    import joblib
    
    # Salvar o modelo treinado com o joblib
    model_filename = 'knn_model.pkl'  # Caminho do arquivo onde o modelo será salvo
    joblib.dump(best_knn, model_filename)
    print(f'Modelosalvo em: {model_filename}')
    data['file'] = model_filename
    return render(request, 'ia_knn_treino.html', data) 

from django.contrib import admin
from django.urls import path
from. import views


def ia_knn_matriz(request):
    import joblib
    from sklearn.metrics import confusion_matrix
    import numpy as np 
    import pandas as pd
    from.models import DatasetCancerBucal  
    Dados_queryset= DatasetCancerBucal.objects.all()
    df= pd.DataFrame(list(Dados_queryset.values()))
    from sklearn.model_selection import train_test_split
    X= df.drop(columns=['grupo', 'id'])
    y= df['grupo']
    model_filename= 'knn_model.pkl'
    best_knn= joblib.load(model_filename)
    y_pred= best_knn.predict(X)
    cm = confusion_matrix(y, y_pred)
    data = {
        'matrix': cm.tolist(),
        'labels': np.unique(y).tolist()
    }
    for i in data['matrix']:
        print(i)
    return render(request, 'ia_knn_matriz.html', data)

def ia_knn_roc(request):
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go
    import numpy as np
    from.models import DatasetCancerBucal
    from django.shortcuts import render
    
    dados_queryset= DatasetCancerBucal.objects.all()
    df= pd.DataFrame(list(dados_queryset.values()))
    X= df.drop(columns=['grupo', 'id'])
    y= df['grupo'].map({'Controle': -1, 'Experimental': 1})
    model_filename = 'knn_model.pkl'
    best_knn = joblib.load(model_filename)
    y_pred_prob = best_knn.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc= auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace (go.Scatter (x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='RandomClassifier',
    line=dict(dash='dash', color='gray')))
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falsos Positivos (FPR)',
        yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
        showlegend=True
    )
    graph= fig.to_html(full_html=False)
    return render(request, 'ia_knn_roc.html', {'graph': graph})

def ia_knn_recall(request):
    import joblib 
    import pandas as pd 
    from sklearn.metrics import precision_recall_curve, auc
    import plotly.graph_objects as go 
    import numpy as np 
    from.models import DatasetCancerBucal
    from django.shortcuts import render 
    
    dados_queryset= DatasetCancerBucal.objects.all()
    df= pd.DataFrame(list(dados_queryset.values()))
    X= df.drop(columns=['grupo', 'id'])
    y= df['grupo'].map({'Controle': -1, 'Experimental': 1})
    model_filename= 'knn_model.pkl'
    best_knn= joblib.load(model_filename)
    y_pred_prob= best_knn.predict_proba(X)[:, 1]
    precision, recall, thresholds= precision_recall_curve(y, y_pred_prob)
    pr_auc= auc(recall, precision)
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'Precision-Recall Curve (AUC = {pr_auc:.2f})', 
    line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode='lines', name='RandomClassifier', 
    line=dict(dash='dash', color='gray')))
    fig.update_layout(
        title='Curva Precision-Recall',
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    graph= fig.to_html(full_html=False)
    return render(request, 'ia_knn_recall.html', {'graph': graph}) 