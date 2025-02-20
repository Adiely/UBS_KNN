# Register your models here.

from django.contrib import admin
from .models import *

@admin.register(DatasetCancerBucal)
class DatasetCancerBucalAdmin(admin.ModelAdmin):
    list_display = ('grupo','idade', 'sexo',  'tabagismo', 'consumo_alcool', 'infeccao_hpv', 'exposicao_solar', 'dieta_inadequada', 'higiene_bucal_inadequada', 'uso_protese_inadequada', 'grau_risco')
    search_fields = ('grupo', 'sexo', 'grau_risco')  # Campos que podem ser pesquisados
    list_filter = ('sexo', 'grau_risco')  # Filtros laterais
