
from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import ListView
from .models import pessoa
#def index(request):
#return render(request, 'index.html')
'''
from django.contrib.auth import authenticate, login, logout
def index(request):
    usuario = request.POST.get('username')
    senha = request.POST.get('password')
    user = authenticate(username=usuario, password=senha)
    if (user is not None):
        login(request, user)
        request.session['username'] = usuario
        request.session['password'] = senha
        request.session['usernamefull'] = user.get_full_name()
        print(request.session['username'])
        print(request.session['password'])
        print(request.session['usernamefull'])
        from django.shortcuts import redirect
        return redirect('pessoa_menu_alias')
    else:
        data ={}
        if (usuario):
           data['msg'] = "Usuário autenticado com sucesso!" + usuario
        return render(request, 'index.html', data)
   
'''   
from django_tables2 import SingleTableView
class pessoa_menu(SingleTableView):
    from .models import pessoa
    from .tables import pessoa_table
    model = pessoa
    table_class = pessoa_table
    template_name_suffix = '_menu'
    table_pagination = {"per_page": 10}
    template_name = 'home/pessoa_menu.html'    
       
def index(request):
    return render(request, 'index.html')

def login_usr(request):
    return render(request,'login_usr.html')

def ver_avisos(request):
    return render(request, 'ver_avisos.html')

def pacientes(request):
    from .models import pessoa
    dicionario = {}
    registros = pessoa.objects.all() 
    dicionario['pessoas'] = registros
    return render(request, 'pacientes.html', dicionario)

from django.urls import reverse_lazy
from django.views.generic.edit import CreateView

class pessoa_create(CreateView):
    from .models import pessoa
    model = pessoa
    fields = ['nome', 'email', 'celular', 'funcao', 'nascimento', 'ativo']
    def get_success_url(self):
        return reverse_lazy('pacientes')
    

from django.views.generic import ListView
class pessoa_list(ListView):
    from .models import pessoa
    model = pessoa
    # Filtro dos dados que serão impresso na lista pessoa
    queryset = pessoa.objects.filter(ativo=True) 
    
    
from django.views.generic.edit import UpdateView
class pessoa_update(UpdateView):
    from .models import pessoa
    model = pessoa
    fields = ['nome', 'email', 'celular', 'funcao', 'nascimento', 'ativo']
    def get_success_url(self):
        return reverse_lazy ('pessoa_list_alias')

from django.views.generic.edit import DeleteView
class pessoa_delete(DeleteView):
    from .models import pessoa
    model = pessoa
    fields = ['nome', 'email', 'celular', 'funcao', 'nascimento', 'ativo']
    template_name_suffix = '_delete'
    def get_success_url(self):
        return reverse_lazy('pessoa_list_alias')
    
def envia_serv(request):
    nome = request.POST.get('nome')
    email = request.POST.get('email')
    celular = request.POST.get('celular')
    funcao = request.POST.get('funcao')
    nascimento = request.POST.get('nascimento')
    ativo = request.POST.get('ativo')
    print("Nome:", nome)
    print("E-mail:", email)
    print("Celular:", celular)
    print("Funçãoo:", funcao)
    print("Nascimento:", nascimento)
    print("ativo:", ativo)
    return render(request, 'envia_serv.html')
    
def get_success_url(self):
    return reverse_lazy('pessoa_menu_alias')

def pacientesorm(request):
    from .models import pessoa
    xnome= request.POST.get('nome')
    xemail= request.POST.get('email')
    xcelular= request.POST.get('celular')
    xfuncao= request.POST.get('funcao')
    xnascimento= request.POST.get('nascimento')
    xativo= request.POST.get('ativo')
    
    print("Nome:", xnome)
    print("eMail:", xemail) 
    print("Celular:", xcelular)
    print("Funcao:", xfuncao)
    print("Nascimento:", xnascimento)
    print("ativo:", xativo)
    
    if (xnome is not None):
        xativo= False
        if(xativo == 'on'):
            xativo= True
        pessoa.objects.create(nome=xnome, email=xemail, celular=xcelular,
                        funcao=xfuncao, nascimento=xnascimento, ativo=xativo)
    return render(request, 'pacientesorm.html')

def exportar(request):
    import pandas as pd
    from.models import pessoa
    eixo_y= []
    p= pessoa.objects.all()
    for _regs in p:
        eixo_x= []
        eixo_x.append(_regs.id)
        eixo_x.append(_regs.nome)
        eixo_x.append(_regs.email)
        eixo_x.append(_regs.celular)
        eixo_x.append(_regs.nascimento)
        eixo_x.append(_regs.ativo)
        eixo_y.append(eixo_x)
    _rotulos_colunas= []
    _rotulos_colunas.append("id")
    _rotulos_colunas.append("nome")
    _rotulos_colunas.append("email")
    _rotulos_colunas.append("celular")
    _rotulos_colunas.append("nascimento")
    _rotulos_colunas.append("ativo")
    df= pd.DataFrame(eixo_y, columns=_rotulos_colunas)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=pessoas.csv'
    df.to_csv(path_or_buf=response)
    return response

def importar(request):
    import pandas as pd
    df=pd.read_csv('/Downloads/pessoas.csv',sep=',')
    for linha, coluna in df.iterrows():
        print(linha, "ID:", coluna['id'])
        print(linha, "Nome:", coluna['nome'])
        print(linha, "eMail:", coluna['email'])
        print(linha, "Celular:", coluna['celular'])
        print(linha, "Nascimento:", coluna['nascimento'])
        print(linha, "Ativo:", coluna['ativo'])
    return HttpResponse("Arquivo Importado")


from .models import Exame
from django.core.files.storage import FileSystemStorage
import os

def import_txt(request):
    if request.method == 'POST' and 'arq_upload' in request.FILES:
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload']
        
        # Salva o arquivo no servidor
        file_path = fss.save(upload.name, upload)
        full_file_path = fss.path(file_path)

        try:
            # Processa o arquivo
            with open(full_file_path, 'r') as file:
                for row in file:
                    # Valida e processa cada linha
                    try:
                        colunas = row.strip().replace("(", "").replace(")", "").split(",")
                        valor = float(colunas[8])  # Index pode variar
                        Exame.objects.create(valor=valor)
                    except (ValueError, IndexError) as e:
                        print(f"Erro ao processar linha: {row} -> {e}")
                        continue

            # Retorna sucesso ao usuário
            return HttpResponse("Arquivo importado com sucesso!")
        
        except Exception as e:
            print(f"Erro ao processar o arquivo: {e}")
            return HttpResponse("Erro ao importar o arquivo.", status=500)
        
        finally:
            # Remove o arquivo do servidor
            if os.path.exists(full_file_path):
                os.remove(full_file_path)

    return render(request, 'import_txt.html')


'''
def importe_txt(request):
    from .models import Exame
    
    import os
    from django.core.files.storage import FileSystemStorage
    if request.method == 'POST' and request.FILES['arq_upload']:
        fss = FileSystemStorage()
        upload = request.FILES['arq_upload']
        file1 = fss.save(upload.name, upload)
        file_url = fss.url(file1)
        print("upload", upload)
        print("file1", file1)
        print("file_url", file_url)
        file2 = open(file1,'r')
        for row in file2:
            colunas = row.replace("(", "").replace(")", "").split(",")
            Exame.objects.create(valor=float(colunas[8]))
        file2.close()
        os.remove(file_url.replace("/", ""))
        return HttpResponse("Arquivo Importado")
    return render(request, 'import_txt.html')

'''