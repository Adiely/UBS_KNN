from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_usr, name='login_usr'),
    path('menu_analitico', views.menu_analitico, name='menu_analitico'),
    path('ia_import', views.ia_import, name='ia_import'),
    path('ia_import_save', views.ia_import_save, name='ia_import_save'),
    path('ia_import_list', views.ia_import_list, name='ia_import_list'),
    path('ia_knn_treino', views.ia_knn_treino, name='ia_knn_treino'),
    path('ia_knn_matriz', views.ia_knn_matriz, name='ia_knn_matriz'),
    path('ia_knn_roc', views.ia_knn_roc, name='ia_knn_roc'),
    path('ia_knn_recall', views.ia_knn_recall, name='ia_knn_recall'),
]