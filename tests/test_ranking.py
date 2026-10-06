import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models.models import Base
from app.services.ranking.elo import calculate_elo_updates
from app.services.ranking.ranking_service import RankingService

@pytest.mark.asyncio
async def test_elo_calculation():
    ratings = {"model_a": 1200.0, "model_b": 1200.0}
    scores = {"model_a": 9.0, "model_b": 5.0}
    
    new_ratings = calculate_elo_updates(ratings, scores)
    
    assert new_ratings["model_a"] > 1200.0
    assert new_ratings["model_b"] < 1200.0
    # Проверка zero-sum (сумма изменений равна нулю)
    diff_a = new_ratings["model_a"] - ratings["model_a"]
    diff_b = new_ratings["model_b"] - ratings["model_b"]
    assert pytest.approx(diff_a + diff_b, abs=1e-5) == 0.0

@pytest.mark.asyncio
async def test_ranking_service_saves_to_db():
    # Создаем ин-мемори SQLite для тестов
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with TestSessionLocal() as session:
        service = RankingService(session)
        
        mock_responses = [
            {
                "model_name": "model_a",
                "text": "Answer A",
                "latency": 0.5,
                "label": "A",
                "correctness": 9.0,
                "clarity": 9.0,
                "conciseness": 9.0,
                "relevance": 9.0,
                "overall": 9.0,
                "is_winner": True
            },
            {
                "model_name": "model_b",
                "text": "Answer B",
                "latency": 0.7,
                "label": "B",
                "correctness": 5.0,
                "clarity": 5.0,
                "conciseness": 5.0,
                "relevance": 5.0,
                "overall": 5.0,
                "is_winner": False
            }
        ]

        evaluation = await service.process_and_save_evaluation(
            prompt="Test prompt",
            judge_model="mistral",
            evaluated_responses=mock_responses
        )

        assert evaluation.id is not None
        
        # Проверим, что лидерборд отдает модели
        leaderboard = await service.repo.get_leaderboard()
        assert len(leaderboard) == 2
        assert leaderboard[0].name == "model_a"  # Победитель выше по Elo
        assert leaderboard[0].wins == 1
        assert leaderboard[1].losses == 1