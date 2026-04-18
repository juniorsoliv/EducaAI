"""
EducaIA - Recomendador de conteúdo educacional.

Estratégia híbrida em duas etapas:
 1. Filtro por categoria prevista pelo classificador de tópicos.
 2. Ranqueamento por similaridade de cosseno entre a consulta e a descrição
    dos materiais, ponderado pelo nível do usuário (iniciante, intermediário,
    avançado).

A similaridade de cosseno sobre vetores TF-IDF é uma técnica amplamente
consolidada para recuperação de informação (MANNING; RAGHAVAN; SCHÜTZE, 2008)
e foi escolhida pela simplicidade e pela ausência de necessidade de treino
supervisionado adicional.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import CATALOGO, Conteudo
from .preprocess import preparar


class RecomendadorConteudo:
    """Recomendador baseado em conteúdo (content-based filtering)."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        descricoes = [preparar(c.descricao + " " + c.titulo) for c in CATALOGO]
        self._matriz = self._vectorizer.fit_transform(descricoes)

    def recomendar(
        self,
        consulta: str,
        categoria: str,
        nivel_usuario: int = 1,
        top_k: int = 3,
    ) -> List[Tuple[Conteudo, float]]:
        """Retorna os top_k conteúdos da categoria mais similares à consulta.

        O nível do usuário é usado para ponderar a pontuação final: materiais
        com nível muito acima do usuário recebem penalização proporcional,
        evitando recomendações excessivamente difíceis.
        """
        vetor_consulta = self._vectorizer.transform([preparar(consulta)])
        similaridades = cosine_similarity(vetor_consulta, self._matriz)[0]

        candidatos: List[Tuple[Conteudo, float]] = []
        for conteudo, sim in zip(CATALOGO, similaridades):
            if conteudo.categoria != categoria:
                continue
            # Penalização suave por diferença de nível.
            diff = max(0, conteudo.nivel - nivel_usuario)
            score = sim * (1.0 - 0.2 * diff)
            candidatos.append((conteudo, float(score)))

        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[:top_k]
