"""
EducaIA - Pré-processamento de texto em português.

Responsabilidades:
 * Normalização (minúsculas, remoção de acentos, pontuação e dígitos).
 * Remoção de stopwords em português.
 * Simplificação textual (substituição de termos técnicos por sinônimos mais
   acessíveis), com o objetivo de apoiar usuários com baixo letramento digital.
"""

import re
import unicodedata
from typing import Iterable


# Lista reduzida de stopwords em português, suficiente para o experimento.
_STOPWORDS = {
    "a", "o", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos",
    "das", "em", "no", "na", "nos", "nas", "por", "pra", "para", "com", "sem",
    "sobre", "que", "qual", "quais", "quando", "como", "onde", "porque", "se",
    "e", "ou", "mas", "não", "nao", "sim", "é", "ser", "foi", "são", "sao",
    "ao", "aos", "pela", "pelo", "pelas", "pelos", "esse", "essa", "esses",
    "essas", "este", "esta", "estes", "estas", "isso", "isto", "aquilo",
    "meu", "minha", "seu", "sua", "tem", "ter", "há", "ha", "já", "ja",
}

# Tabela de simplificação (termo técnico -> expressão acessível).
_SIMPLIFICACAO = {
    "criptografia": "proteção por código",
    "autenticação": "confirmação de identidade",
    "phishing": "golpe por mensagem falsa",
    "ransomware": "vírus que sequestra dados",
    "machine learning": "aprendizado do computador",
    "deep learning": "aprendizado profundo do computador",
    "rede neural": "sistema que imita o cérebro",
    "algoritmo": "passo a passo que o computador segue",
    "lgpd": "lei geral de proteção de dados pessoais",
    "vpn": "conexão privada segura",
    "dns": "serviço que traduz endereços da internet",
    "tcp ip": "regras que os computadores usam para conversar na internet",
}


def _strip_accents(texto: str) -> str:
    """Remove acentuação preservando a base ASCII das palavras."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalizar(texto: str) -> str:
    """Normaliza o texto: caixa baixa, sem acentos, pontuação ou dígitos."""
    texto = texto.lower()
    texto = _strip_accents(texto)
    texto = re.sub(r"[^a-z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def remover_stopwords(tokens: Iterable[str]) -> list:
    """Filtra tokens removendo stopwords em português."""
    return [t for t in tokens if t and t not in _STOPWORDS]


def preparar(texto: str) -> str:
    """Pipeline completo de pré-processamento retornando string normalizada."""
    norm = normalizar(texto)
    tokens = remover_stopwords(norm.split())
    return " ".join(tokens)


def simplificar(texto: str) -> str:
    """Substitui termos técnicos por expressões acessíveis.

    Essa simplificação é propositalmente rasa (dicionário + substring match).
    O objetivo é demonstrar a viabilidade da técnica; em produção, o módulo
    deverá ser substituído por modelo de paráfrase controlada.
    """
    resultado = texto
    for termo, equivalente in _SIMPLIFICACAO.items():
        padrao = re.compile(rf"\b{re.escape(termo)}\b", flags=re.IGNORECASE)
        resultado = padrao.sub(equivalente, resultado)
    return resultado
