"""
EducaIA - Classificador de tópicos educacionais.

Implementa um pipeline clássico de PLN (Processamento de Linguagem Natural):
pré-processamento -> vetorização TF-IDF -> Multinomial Naive Bayes.

A escolha do Naive Bayes se justifica por três critérios:
 1. Desempenho reconhecido em classificação de texto curto (MANNING; RAGHAVAN;
    SCHÜTZE, 2008).
 2. Baixo custo computacional, viabilizando execução em ambientes de baixa
    capacidade — alinhado ao recorte de inclusão digital do trabalho.
 3. Interpretabilidade: permite inspecionar as probabilidades atribuídas a
    cada classe, o que é relevante para auditoria pedagógica.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import TRAIN_DATA
from .preprocess import preparar


class ClassificadorTopicos:
    """Classificador de tópicos educacionais com TF-IDF + MultinomialNB."""

    def __init__(self) -> None:
        self._pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("nb", MultinomialNB(alpha=0.3)),
        ])
        self._treinado = False

    def treinar(self) -> None:
        """Treina o pipeline com o dataset sintético em português."""
        textos = [preparar(t) for t, _ in TRAIN_DATA]
        rotulos = [r for _, r in TRAIN_DATA]
        self._pipeline.fit(textos, rotulos)
        self._treinado = True

    def prever(self, texto: str) -> Tuple[str, float]:
        """Retorna (rótulo previsto, confiança)."""
        if not self._treinado:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes.")
        vetor = [preparar(texto)]
        rotulo = self._pipeline.predict(vetor)[0]
        proba = float(max(self._pipeline.predict_proba(vetor)[0]))
        return rotulo, proba

    def ranking(self, texto: str) -> List[Tuple[str, float]]:
        """Retorna lista ordenada (rótulo, probabilidade) para auditoria."""
        if not self._treinado:
            raise RuntimeError("Modelo não treinado. Chame treinar() antes.")
        vetor = [preparar(texto)]
        classes = self._pipeline.named_steps["nb"].classes_
        probs = self._pipeline.predict_proba(vetor)[0]
        pares = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        return [(c, float(p)) for c, p in pares]
