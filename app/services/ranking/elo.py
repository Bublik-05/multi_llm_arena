from typing import Dict, List, Tuple


def calculate_elo_updates(
    ratings: Dict[str, float],
    scores: Dict[str, float],
    k: float = 32.0,
    draw_threshold: float = 0.05,
) -> Dict[str, float]:
    """Вычисляет новые значения Elo для набора моделей на основе попарных сравнений.
    
    ratings: словарь {model_name: current_elo}
    scores: словарь {model_name: overall_score}
    """
    models = list(ratings.keys())
    n = len(models)
    if n < 2:
        return ratings.copy()

    new_ratings = ratings.copy()
    deltas = {m: 0.0 for m in models}

    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = models[i], models[j]
            r1, r2 = ratings[m1], ratings[m2]
            s1, s2 = scores[m1], scores[m2]

            # Ожидаемый результат (Expected score)
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))

            # Фактический результат (Actual score) с учетом порога ничьей
            diff = s1 - s2
            if abs(diff) < draw_threshold:
                a1, a2 = 0.5, 0.5
            elif diff > 0:
                a1, a2 = 1.0, 0.0
            else:
                a1, a2 = 0.0, 1.0

            # Накопление дельт (делим на n - 1 по условию)
            deltas[m1] += k * (a1 - e1) / (n - 1)
            deltas[m2] += k * (a2 - e2) / (n - 1)

    for m in models:
        new_ratings[m] = ratings[m] + deltas[m]

    return new_ratings