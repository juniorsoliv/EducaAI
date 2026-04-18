"""
EducaIA - Módulo de dados sintéticos para treinamento e catálogo educacional.

Este módulo contém um conjunto rotulado de perguntas/descrições educacionais em
português, cobrindo categorias iniciais de tecnologia. Os dados servem para
treinar o classificador de tópicos e alimentar o recomendador de conteúdo.

Autor: Marcos (TCC - Ciência da Computação - UNIP, 2026).
"""

from dataclasses import dataclass
from typing import List


# -----------------------------------------------------------------------------
# Dataset de treinamento do classificador de tópicos (Naive Bayes + TF-IDF).
# Cada tupla contém (texto, rótulo). A escolha por um dataset manualmente
# curado em português busca mitigar vieses de tradução automática.
# -----------------------------------------------------------------------------
TRAIN_DATA: List[tuple] = [
    # ---- Programação ----
    ("como declarar uma variável em python", "programacao"),
    ("o que é uma função em python", "programacao"),
    ("diferença entre lista e tupla em python", "programacao"),
    ("como fazer um loop for em java", "programacao"),
    ("o que é programação orientada a objetos", "programacao"),
    ("como usar classes e objetos em java", "programacao"),
    ("como instalar dependências com pip", "programacao"),
    ("como depurar um programa em c", "programacao"),
    ("o que é um algoritmo de ordenação", "programacao"),
    ("diferença entre pilha e fila", "programacao"),
    ("o que é recursão em programação", "programacao"),
    ("como criar uma api rest em python", "programacao"),

    # ---- Redes e Internet ----
    ("o que é o protocolo tcp ip", "redes"),
    ("como funciona o dns na internet", "redes"),
    ("diferença entre http e https", "redes"),
    ("o que é um endereço ip", "redes"),
    ("como funciona uma rede local", "redes"),
    ("o que é largura de banda", "redes"),
    ("o que é um roteador e um switch", "redes"),
    ("o que é uma vpn", "redes"),

    # ---- Segurança e Privacidade ----
    ("como criar uma senha segura", "seguranca"),
    ("o que é criptografia de ponta a ponta", "seguranca"),
    ("o que é phishing", "seguranca"),
    ("como proteger minha conta de email", "seguranca"),
    ("o que é autenticação de dois fatores", "seguranca"),
    ("o que é a lgpd lei geral de proteção de dados", "seguranca"),
    ("como funciona um ataque de ransomware", "seguranca"),

    # ---- Inteligência Artificial ----
    ("o que é inteligência artificial", "inteligencia_artificial"),
    ("como funciona o aprendizado de máquina", "inteligencia_artificial"),
    ("diferença entre ia e machine learning", "inteligencia_artificial"),
    ("o que é uma rede neural", "inteligencia_artificial"),
    ("o que é processamento de linguagem natural", "inteligencia_artificial"),
    ("como o chatgpt funciona", "inteligencia_artificial"),
    ("o que é aprendizado profundo deep learning", "inteligencia_artificial"),
    ("como treinar um modelo de classificação", "inteligencia_artificial"),
    ("o que é viés algorítmico", "inteligencia_artificial"),

    # ---- Inclusão Digital e Acessibilidade ----
    ("como usar um leitor de tela", "inclusao_digital"),
    ("o que é letramento digital", "inclusao_digital"),
    ("como aprender a usar o computador do zero", "inclusao_digital"),
    ("como acessar serviços públicos pela internet", "inclusao_digital"),
    ("o que é inclusão digital", "inclusao_digital"),
    ("como baixar e usar o aplicativo do governo", "inclusao_digital"),
    ("como fazer uma pesquisa simples no google", "inclusao_digital"),
    ("o que é tecnologia assistiva", "inclusao_digital"),
    ("como ajustar o tamanho da letra no celular", "inclusao_digital"),
    ("o que é acessibilidade digital", "inclusao_digital"),
]


# -----------------------------------------------------------------------------
# Catálogo de materiais educacionais (conteúdo recomendável).
# Atributos: id, título, categoria, nível (1-iniciante, 2-intermediário, 3-avançado),
# descrição (usada no recomendador baseado em conteúdo) e URL de referência.
# -----------------------------------------------------------------------------
@dataclass
class Conteudo:
    id: int
    titulo: str
    categoria: str
    nivel: int
    descricao: str
    url: str


CATALOGO: List[Conteudo] = [
    Conteudo(1, "Introdução ao Python para iniciantes", "programacao", 1,
             "Curso passo a passo sobre variáveis, tipos, entrada e saída e estruturas de decisão em Python.",
             "https://cursoemvideo.com/curso/python/"),
    Conteudo(2, "Estruturas de dados em Java", "programacao", 2,
             "Listas, pilhas, filas, árvores e tabelas hash aplicadas à linguagem Java.",
             "https://cursoemvideo.com/curso/java/"),
    Conteudo(3, "Algoritmos e lógica de programação", "programacao", 1,
             "Fundamentos de lógica, condicionais, laços e funções com exercícios progressivos.",
             "https://cursoemvideo.com/curso/logica/"),
    Conteudo(4, "Fundamentos de redes de computadores", "redes", 1,
             "Modelo TCP/IP, endereçamento IP, DNS, HTTP, HTTPS e conceitos básicos de segurança de rede.",
             "https://www.cisco.com/c/pt_br/training-events/netacad.html"),
    Conteudo(5, "Segurança digital para leigos", "seguranca", 1,
             "Boas práticas de senhas, autenticação em duas etapas, phishing e proteção de dados pessoais.",
             "https://cartilha.cert.br/"),
    Conteudo(6, "Lei Geral de Proteção de Dados (LGPD) na prática", "seguranca", 2,
             "Princípios, bases legais, direitos dos titulares e adequação organizacional à LGPD.",
             "https://www.gov.br/anpd/"),
    Conteudo(7, "Introdução à Inteligência Artificial", "inteligencia_artificial", 1,
             "Conceitos de IA, aprendizado de máquina, redes neurais e aplicações práticas em linguagem acessível.",
             "https://www.ibm.com/br-pt/topics/artificial-intelligence"),
    Conteudo(8, "Aprendizado de máquina com scikit-learn", "inteligencia_artificial", 2,
             "Pipelines, vetorização de texto com TF-IDF, classificadores Naive Bayes, SVM e avaliação de modelos.",
             "https://scikit-learn.org/stable/"),
    Conteudo(9, "Ética e viés em Inteligência Artificial", "inteligencia_artificial", 3,
             "Discussão sobre vieses algorítmicos, explicabilidade, privacidade e regulação de sistemas de IA.",
             "https://unesdoc.unesco.org/ark:/48223/pf0000380455"),
    Conteudo(10, "Alfabetização digital para adultos", "inclusao_digital", 1,
             "Introdução ao uso de computadores, smartphones, e-mail e serviços digitais do governo.",
             "https://www.gov.br/mcom/pt-br"),
    Conteudo(11, "Tecnologias assistivas e acessibilidade", "inclusao_digital", 2,
             "Leitores de tela, legendagem automática, reconhecimento de voz e diretrizes WCAG para web.",
             "https://www.w3.org/WAI/fundamentals/accessibility-intro/pt-br"),
    Conteudo(12, "Navegando com segurança na internet", "inclusao_digital", 1,
             "Como identificar golpes, proteger dados pessoais e usar redes sociais com consciência.",
             "https://cartilha.cert.br/fasciculos/"),
]
