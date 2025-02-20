
from django.contrib import admin
from django.urls import path 
from . import views
 
urlpatterns = [
    path('', views.index, name='index_alias'),
    path('home', views.index, name='index'), 
    path('login_usr', views.login_usr, name='login_usr_alias'),  
    path('avisos', views.ver_avisos, name='ver_avisos'),
    path('pacientes', views.pacientes, name='pacientes'),
    path("pessoa_create/", views.pessoa_create.as_view(), name='pessoa_create_alias'),
    path("pessoa_list/", views.pessoa_list.as_view(), name='pessoa_list_alias'),
    path("pessoa_update/<int:pk>/", views.pessoa_update.as_view(),name='pessoa_update_alias'),
    path('pessoa_delete/<int:pk>/', views.pessoa_delete.as_view(),name='pessoa_delete_alias'),
    path('envia_serv', views.envia_serv, name='envia_serv_alias'),
    path('menu', views.pessoa_menu.as_view(), name='pessoa_menu_alias'),
    
    path('pacientesorm', views.pacientesorm, name='pacientesorm_alias'),
    path('exportar', views.exportar, name='exportar_alias'),
    #path('importar', views.importar, name='importar_alias'),
    path('import_txt', views.import_txt, name='import_txt_alias'),
]


