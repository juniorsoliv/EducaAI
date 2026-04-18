"""
Script de execução do experimento do protótipo EducaIA.

Roda um conjunto de consultas representativas, exibe o tópico previsto,
a confiança do modelo, a versão simplificada da consulta e a lista de
conteúdos recomendados. Os resultados são gravados em JSON para fins de
reprodutibilidade e inclusão como evidência no TCC.
"""

from __future__ import annotations

import json
from pathlib import Path

from educaia.service import ServicoEducaIA


CONSULTAS_TESTE = [
    ("minha avó quer aprender a usar o celular, por onde começar?", 1),
    ("o que é inteligência artificial e como funciona na prática?", 1),
    ("como proteger minha conta de email contra phishing?", 2),
    ("qual a diferença entre uma lista e uma tupla em python?", 2),
    ("como funciona o protocolo tcp ip em uma rede local?", 2),
    ("o que é viés algorítmico em sistemas de aprendizado de máquina?", 3),
]


def main() -> None:
    servico = ServicoEducaIA()
    relatorio = []

    for consulta, nivel in CONSULTAS_TESTE:
        resposta = servico.atender_dict(consulta, nivel_usuario=nivel, top_k=3)
        relatorio.append(resposta)

        print("=" * 78)
        print(f"Consulta: {consulta}")
        print(f"Nível do usuário: {nivel}")
        print(f"Consulta simplificada: {resposta['consulta_simplificada']}")
        print(
            f"Tópico previsto: {resposta['topico_previsto']} "
            f"(confiança={resposta['confianca_topico']:.2%})"
        )
        print("Recomendações:")
        for rec in resposta["recomendacoes"]:
            print(
                f"  - [{rec['nivel']}] {rec['titulo']} "
                f"(score={rec['score']:.3f}) -> {rec['url']}"
            )
        print()

    saida = Path(__file__).resolve().parent.parent / "evidencias" / "resultados.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório salvo em: {saida}")


if __name__ == "__main__":
    main()
