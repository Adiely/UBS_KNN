# Generated by Django 5.1.4 on 2025-01-05 14:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Pessoa',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nome', models.CharField(max_length=50, verbose_name='Nome')),
                ('email', models.CharField(max_length=50, verbose_name='Email')),
                ('celular', models.CharField(blank=True, max_length=20, null=True, verbose_name='celular')),
                ('funcao', models.CharField(blank=True, max_length=30, null=True, verbose_name='Funcao')),
                ('nascimento', models.DateField(blank=True, null=True, verbose_name='Nascimento')),
                ('ativo', models.BooleanField(default=True, verbose_name='Ativo')),
            ],
            options={
                'ordering': ['nome', 'ativo'],
            },
        ),
    ]
