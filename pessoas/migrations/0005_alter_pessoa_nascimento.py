# Generated by Django 5.1.4 on 2025-01-05 16:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pessoas', '0004_pessoa_nascimento'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pessoa',
            name='nascimento',
            field=models.DateField(blank=True, null=True, verbose_name='Nascimento'),
        ),
    ]
