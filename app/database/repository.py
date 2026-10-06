from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any, Optional
from app.models.models import ModelRecord, Evaluation, Response, RatingHistory

class EvaluationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_model(self, name: str, provider: str = "ollama") -> ModelRecord:
        result = await self.session.execute(select(ModelRecord).filter_by(name=name))
        model = result.scalars().first()
        if not model:
            model = ModelRecord(name=name, provider=provider, elo=1200.0)
            self.session.add(model)
            await self.session.flush()
        return model

    async def save_evaluation_result(
        self,
        prompt: str,
        judge_model: str,
        results_data: List[Dict[str, Any]],
        updated_elos: Dict[str, float]
    ) -> Evaluation:
        """Сохраняет полный результат оценки (Evaluation, Response, обновление Elo и историю) одной транзакцией"""
        
        # 1. Создаем запись Evaluation
        evaluation = Evaluation(prompt=prompt, judge_model=judge_model)
        self.session.add(evaluation)
        await self.session.flush() # чтобы получить evaluation.id

        # 2. Определяем победителя (у кого больше overall или is_winner)
        # Сохраняем ответы
        for res in results_data:
            model_name = res["model_name"]
            
            # Убедимся, что модель есть в таблице models
            model_rec = await self.get_or_create_model(model_name)

            # Создаем ответ
            response = Response(
                evaluation_id=evaluation.id,
                model_name=model_name,
                text=res["text"],
                latency=res["latency"],
                label=res["label"],
                correctness=res.get("correctness"),
                clarity=res.get("clarity"),
                conciseness=res.get("conciseness"),
                relevance=res.get("relevance"),
                overall=res.get("overall"),
                is_winner=1 if res.get("is_winner") else 0,
                judge_raw=res.get("judge_raw")
            )
            self.session.add(response)

            # Обновляем статистику модели
            model_rec.evaluations_count += 1
            model_rec.latency_sum += res["latency"]
            
            if res.get("overall") is not None:
                model_rec.score_sum += res["overall"]
            
            if res.get("is_error"):
                model_rec.errors += 1
            elif res.get("is_winner"):
                model_rec.wins += 1
            else:
                model_rec.losses += 1

            # Обновляем Elo, если он пересчитан
            if model_name in updated_elos:
                model_rec.elo = updated_elos[model_name]
                
                # Пишем историю изменения Elo
                history = RatingHistory(
                    model_name=model_name,
                    elo=model_rec.elo,
                    evaluation_id=evaluation.id
                )
                self.session.add(history)

        await self.session.commit()
        return evaluation

    async def get_leaderboard(self) -> List[ModelRecord]:
        """Возвращает список моделей, отсортированный по Elo (по убыванию)"""
        result = await self.session.execute(
            select(ModelRecord).order_by(ModelRecord.elo.desc())
        )
        return result.scalars().all()