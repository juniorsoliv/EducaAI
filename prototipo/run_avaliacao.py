"""
Avaliação quantitativa do classificador de tópicos do EducaIA.

Utiliza validação cruzada estratificada (k=5) e também um conjunto de teste
manualmente elaborado, externo ao treinamento, para reportar acurácia,
precision/recall/F1 por classe e matriz de confusão.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from educaia.classifier import ClassificadorTopicos
from educaia.data import TRAIN_DATA
from educaia.preprocess import preparar


TESTE_EXTERNO = [
    ("como faço para criar uma função em python", "programacao"),
    ("o que é orientação a objetos", "programacao"),
    ("como descobrir o endereço ip do meu computador", "redes"),
    ("o que é um protocolo seguro na internet", "redes"),
    ("como evitar cair em golpes de phishing", "seguranca"),
    ("meus dados pessoais estão protegidos pela lgpd", "seguranca"),
    ("o que é aprendizado de máquina supervisionado", "inteligencia_artificial"),
    ("como funciona uma rede neural artificial", "inteligencia_artificial"),
    ("minha mãe está aprendendo a usar o whatsapp", "inclusao_digital"),
    ("como aumentar a letra no celular dos idosos", "inclusao_digital"),
]


def cross_validate() -> dict:
    textos = np.array([preparar(t) for t, _ in TRAIN_DATA])
    rotulos = np.array([r for _, r in TRAIN_DATA])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acuracias = []
    for treino_idx, val_idx in skf.split(textos, rotulos):
        clf = ClassificadorTopicos()
        # Injetamos dataset parcial reaproveitando o pipeline interno.
        clf._pipeline.fit(textos[treino_idx], rotulos[treino_idx])
        clf._treinado = True
        preds = clf._pipeline.predict(textos[val_idx])
        acuracias.append(accuracy_score(rotulos[val_idx], preds))
    return {
        "folds": 5,
        "acuracia_media": float(np.mean(acuracias)),
        "desvio_padrao": float(np.std(acuracias)),
        "acuracia_por_fold": [float(a) for a in acuracias],
    }


def teste_externo() -> dict:
    clf = ClassificadorTopicos()
    clf.treinar()
    preds = [clf.prever(t)[0] for t, _ in TESTE_EXTERNO]
    y_true = [r for _, r in TESTE_EXTERNO]

    labels = sorted(set(y_true) | set(preds))
    cm = confusion_matrix(y_true, preds, labels=labels).tolist()
    report = classification_report(y_true, preds, zero_division=0, output_dict=True)
    return {
        "acuracia": float(accuracy_score(y_true, preds)),
        "matriz_confusao": {"labels": labels, "matriz": cm},
        "relatorio": report,
        "previsoes": [
            {"texto": t, "esperado": e, "previsto": p}
            for (t, e), p in zip(TESTE_EXTERNO, preds)
        ],
    }


def main() -> None:
    resultado = {
        "validacao_cruzada": cross_validate(),
        "teste_externo": teste_externo(),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))

    saida = Path(__file__).resolve().parent.parent / "evidencias" / "avaliacao.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAvaliação salva em: {saida}")


if __name__ == "__main__":
    main()
