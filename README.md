# Sistema Gerenciador de UBS

## Aplicação Web com Django e Machine Learning (KNN)

Este projeto foi desenvolvido como parte da disciplina **Framework** da pós-graduação em **Sistemas e Agentes Inteligentes** da **UFG**. A aplicação web foi construída utilizando o **Django Framework** e tem como objetivo acessar dados de um banco de dados, realizar o treinamento de um modelo de machine learning **KNN (K-nearest neighbors)** e fornecer ferramentas para avaliar seu desempenho.

## Alunos
Samantha Adiely Alecrim
Edson Laranjeiras
Billy Fádel

## Funcionalidades

- **Importação e Exportação de Dados**: Opção para importar/exportar arquivos CSV para o banco de dados.
- **Treinamento de Modelo**: Implementação de um modelo de machine learning (**KNN**) treinado com os dados importados.
- **Métricas de Avaliação**:
  - Curva ROC
  - Precisão e Recall
  - Matriz de Confusão

## Tecnologias Utilizadas

- **Linguagem**: Python
- **Framework**: Django
- **Banco de Dados**: SQLite
- **Machine Learning**: Scikit-Learn
- **Front-end**: HTML (Bootstrap)

## Instalação e Execução

### Pré-requisitos

- Python 3 instalado
- Virtualenv instalado (opcional, mas recomendado)

### Passo a Passo

1. **Clone o repositório**  
   ```sh
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio

Crie e ative um ambiente virtual
- venv\Scripts\activate #No Windowns

Instale as dependências
- pip install -r requirements.txt

Realize as migrações do banco de dados
- python manage.py migrate

Inicie o servidor Django
- python manage.py runserver

Como Utilizar:

1. Importe um arquivo CSV para carregar os dados no banco de dados.
2. Treine o modelo KNN utilizando os dados importados.
3. Visualize as métricas de avaliação, como a Curva ROC, Precisão, Recall e Matriz de Confusão.
4. Exporte os dados processados em formato CSV.
