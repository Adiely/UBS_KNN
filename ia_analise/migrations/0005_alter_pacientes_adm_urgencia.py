# Generated by Django 4.2.17 on 2025-01-30 11:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ia_analise', '0004_rename_adm_urgente_pacientes_adm_urgencia'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pacientes',
            name='Adm_Urgencia',
            field=models.FloatField(blank=True, null=True, verbose_name='Adm_Urgencia'),
        ),
    ]
