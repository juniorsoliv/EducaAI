# Descrição Técnica (Português)

O EducaIA é um protótipo de assistente educacional inteligente que utiliza técnicas de Processamento de Linguagem Natural (PLN) e aprendizado de máquina para interpretar dúvidas em linguagem natural e recomendar conteúdos educacionais personalizados.

A arquitetura do sistema é baseada em um pipeline modular composto por três etapas principais:

### Pré-processamento de texto

A entrada do usuário (consulta em linguagem natural) é normalizada e simplificada por meio de técnicas de limpeza textual, preparando os dados para análise computacional.

### Classificação de tópicos

O sistema utiliza um modelo de machine learning baseado em:

- Vetorização TF-IDF (Term Frequency–Inverse Document Frequency)
- Algoritmo Multinomial Naive Bayes

Esse pipeline permite classificar a consulta do usuário em um tópico educacional específico. O modelo é treinado com um conjunto de dados sintético em português e retorna:

- O tópico previsto
- Um nível de confiança associado à previsão

A escolha do Naive Bayes se dá pelo seu baixo custo computacional, boa performance em textos curtos e interpretabilidade, tornando-o adequado para ambientes com recursos limitados.

### Sistema de recomendação de conteúdo

Após a classificação, um módulo de recomendação sugere conteúdos educacionais relevantes com base em:

- Tópico identificado
- Nível de conhecimento do usuário
- Similaridade entre a consulta e os conteúdos disponíveis

O sistema retorna uma lista ranqueada de recomendações contendo:

- Título
- Categoria
- Nível
- Descrição
- URL
- Score de relevância

### Camada de Serviço (Orquestração)

O núcleo do sistema é a classe ServicoEducaIA, que atua como uma fachada responsável por integrar todos os componentes. Ela executa o fluxo completo:

Entrada → Classificação → Recomendação → Resposta estruturada

A resposta final é encapsulada em um objeto estruturado contendo:

- Consulta original e simplificada
- Tópico previsto e confiança
- Lista de recomendações personalizadas

### Objetivo

O EducaIA foi projetado com foco em acessibilidade educacional, permitindo que usuários obtenham trilhas de aprendizado a partir de perguntas simples, mesmo em ambientes com baixa capacidade computacional.

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Technical Description (English)

EducaIA is a prototype of an intelligent educational assistant that leverages Natural Language Processing (NLP) and machine learning techniques to interpret user queries in natural language and recommend personalized learning content.

The system architecture is built around a modular pipeline composed of three main stages:

### Text Preprocessing

User input is cleaned and simplified through text normalization techniques, preparing it for computational analysis.

### Topic Classification

The system uses a machine learning pipeline based on:

- TF-IDF (Term Frequency–Inverse Document Frequency) vectorization
- Multinomial Naive Bayes algorithm

This pipeline classifies the user query into a specific educational topic. The model is trained on a synthetic dataset in Portuguese and outputs:

- Predicted topic
- Confidence score

Naive Bayes was chosen due to its low computational cost, strong performance on short texts, and interpretability, making it suitable for low-resource environments.

### Content Recommendation System

After classification, a recommendation module suggests relevant educational resources based on:

- Identified topic
- User knowledge level
- Similarity between the query and available content

The system returns a ranked list of recommendations, including:

- Title
- Category
- Level
- Description
- URL
- Relevance score

### Service Layer (Orchestration)

The core of the system is the ServicoEducaIA class, which acts as a façade that integrates all components and executes the full pipeline:

Input → Classification → Recommendation → Structured Response

The final output is encapsulated in a structured object containing:

- Original and simplified query
- Predicted topic and confidence
- Personalized recommendations list

### Purpose

EducaIA is designed with a focus on educational accessibility, enabling users to obtain learning paths from simple natural language questions, even in low-computational environments.
