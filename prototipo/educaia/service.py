"""
EducaIA - Serviço de orquestração (caso de uso principal).

Combina classificador de tópicos e recomendador de conteúdo em um único
caso de uso: "recomendar trilha de aprendizagem a partir de uma dúvida em
linguagem natural".
"""

from dataclasses import dataclass, asdict
from typing import List

from .classifier import ClassificadorTopicos
from .preprocess import simplificar
from .recommender import RecomendadorConteudo


@dataclass
class RespostaEducaIA:
    consulta_original: str
    consulta_simplificada: str
    topico_previsto: str
    confianca_topico: float
    nivel_usuario: int
    recomendacoes: List[dict]


class ServicoEducaIA:
    """Fachada do protótipo. Instancia modelos uma única vez."""

    def __init__(self) -> None:
        self._classificador = ClassificadorTopicos()
        self._classificador.treinar()
        self._recomendador = RecomendadorConteudo()

    def atender(
        self,
        consulta: str,
        nivel_usuario: int = 1,
        top_k: int = 3,
    ) -> RespostaEducaIA:
        topico, confianca = self._classificador.prever(consulta)
        recomendacoes = self._recomendador.recomendar(
            consulta=consulta,
            categoria=topico,
            nivel_usuario=nivel_usuario,
            top_k=top_k,
        )
        return RespostaEducaIA(
            consulta_original=consulta,
            consulta_simplificada=simplificar(consulta),
            topico_previsto=topico,
            confianca_topico=round(confianca, 4),
            nivel_usuario=nivel_usuario,
            recomendacoes=[
                {
                    "id": c.id,
                    "titulo": c.titulo,
                    "categoria": c.categoria,
                    "nivel": c.nivel,
                    "descricao": c.descricao,
                    "url": c.url,
                    "score": round(score, 4),
                }
                for c, score in recomendacoes
            ],
        )

    def atender_dict(self, consulta: str, nivel_usuario: int = 1, top_k: int = 3) -> dict:
        return asdict(self.atender(consulta, nivel_usuario, top_k))
