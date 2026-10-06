from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from app.services.ranking.elo import calculate_elo_updates
from app.database.repository import EvaluationRepository

class RankingService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = EvaluationRepository(session)

    async def process_and_save_evaluation(
        self,
        prompt: str,
        judge_model: str,
        evaluated_responses: List[Dict[str, Any]]
    ):
        """
        Принимает результаты судейства моделей, подгружает текущие Elo из базы,
        считает новые Elo через elo.py и сохраняет всё в БД.
        """
        # 1. Получаем текущие Elo всех участвующих моделей из базы
        current_ratings = {}
        scores = {}

        for res in evaluated_responses:
            model_name = res["model_name"]
            model_rec = await self.repo.get_or_create_model(model_name)
            current_ratings[model_name] = model_rec.elo
            scores[model_name] = res.get("overall", 0.0)

        # 2. Считаем новые Elo с помощью функции calculate_elo_updates
        updated_elos = calculate_elo_updates(current_ratings, scores)

        # 3. Сохраняем всё в репозиторий (включая историю и статистику)
        evaluation = await self.repo.save_evaluation_result(
            prompt=prompt,
            judge_model=judge_model,
            results_data=evaluated_responses,
            updated_elos=updated_elos
        )

        return evaluation