'''O modelo Acessa o banco de dados
e retorna os dados necessário para 
atender as necessidades da views
'''

from django.db import models
#Criando uma class pessoa
'''
class pessoa (models.Model):
    nome = models.CharField(max_length=50, null=False, blank=False, verbose_name='Nome')
    email = models.CharField(max_length=50, null=False, blank=False, verbose_name='Email')
    celular = models.CharField(max_length=20, null=True, blank=True, verbose_name='celular')
    funcao = models.CharField(max_length=30, null=True, blank=True, verbose_name='Funcao')
    nascimento = models.DateField(null=True, blank=True,verbose_name='Nascimento')
    ativo = models.BooleanField(default=True, verbose_name='Ativo')
    
def __str__(self):
             return self.nome
class Meta:
        ordering = ['nome','ativo',] # O Sinal - ordenar ascedente ou Descendente
        
from django.db import models
'''

class pessoa(models.Model):
    nome = models.CharField(max_length=50, null=False, blank=False, verbose_name='Nome')
    email = models.CharField(max_length=50, null=False, blank=False, verbose_name='eMail')
    celular = models.CharField(max_length=20, null=True, blank=True, verbose_name='celular')
    funcao = models.CharField(max_length=30, null=True, blank=True, verbose_name='Funcao')
    nascimento = models.DateField(null=True, blank=True, verbose_name='Nascimento')
    ativo = models.BooleanField(default=True, verbose_name='Ativo')

def __str__(self):
        return self.nome
class Meta:
        ordering = ['nome', 'funcao']

        
class procedimento(models.Model):
    descricao = models.CharField(max_length=50, null=False, blank=False, verbose_name='Descricao')
    cid = models.CharField(max_length=20, null=False, blank=False, verbose_name='CID')
    valor = models.FloatField(null=True, blank=True, default=None, verbose_name='Valor')

    def __str__(self): 
        return self.descricao + str(self.valor)

    class Meta:
        ordering = ['descricao']
        
class procedimento_executado(models.Model):
    pessoa = models.ForeignKey(pessoa, on_delete=models.CASCADE)
    procedimento = models.ForeignKey(procedimento, on_delete=models.CASCADE)
    obs = models.CharField(max_length=50, null=False, blank=False, verbose_name='Obs')
    quantidade = models.FloatField(null=True, blank=True, default=None, verbose_name='Quantidade')

    def __str__(self): 
        return self.pessoa.nome + " - " + self.procedimento.descricao + " - " + self.obs

    class Meta:
        ordering = ['pessoa', 'procedimento']
        
class Exame(models.Model):
    valor = models.FloatField(
        null=True, 
        blank=True, 
        default=None,
        verbose_name='Valor'
    )

    def __str__(self):
        return str(self.valor)
