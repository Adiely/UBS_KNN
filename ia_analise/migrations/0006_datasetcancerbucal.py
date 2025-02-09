# Generated by Django 4.2.18 on 2025-02-08 22:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ia_analise', '0005_alter_pacientes_adm_urgencia'),
    ]

    operations = [
        migrations.CreateModel(
            name='DatasetCancerBucal',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('grupo', models.CharField(blank=True, max_length=50, null=True, verbose_name='Grupo')),
                ('tabagismo', models.CharField(blank=True, max_length=50, null=True, verbose_name='Tabagismo')),
                ('consumo_de_alcool', models.CharField(blank=True, max_length=50, null=True, verbose_name='Consumo de Álcool')),
                ('idade', models.FloatField(blank=True, null=True, verbose_name='Idade')),
                ('sexo', models.CharField(blank=True, max_length=50, null=True, verbose_name='Sexo')),
                ('infeccao_por_hpv', models.CharField(blank=True, max_length=50, null=True, verbose_name='Infecção por HPV')),
                ('exposicao_solar', models.CharField(blank=True, max_length=50, null=True, verbose_name='Exposição Solar')),
                ('dieta_inadequada', models.CharField(blank=True, max_length=50, null=True, verbose_name='Dieta Inadequada')),
                ('higiene_bucal_inadequada', models.CharField(blank=True, max_length=50, null=True, verbose_name='Higiene Bucal Inadequada')),
                ('uso_de_protese_inadequada', models.CharField(blank=True, max_length=50, null=True, verbose_name='Uso de Prótese Inadequada')),
                ('grau_de_risco', models.CharField(blank=True, max_length=50, null=True, verbose_name='Grau de Risco')),
            ],
        ),
    ]
