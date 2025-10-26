import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardRemove
from PIL import Image

from app.config import CollageConfig
from app.fsm import TryOnStates, setup_router
from app.models import GlassModel
from app.services.collage import CollageProcessingError, CollageSourceUnavailable
from app.services.recommendation import RecommendationResult
from app.texts import messages as msg


@dataclass
class PhotoStub:
    file_unique_id: str


class DummyBot:
    def __init__(self) -> None:
        self.deleted: list[tuple[int, int]] = []
        self.downloads: list[tuple[Any, Path]] = []

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        self.deleted.append((chat_id, message_id))

    async def download(self, photo: PhotoStub, destination: Path) -> None:
        self.downloads.append((photo, destination))


class DummyMessage:
    def __init__(self, user_id: int, bot: DummyBot, *, message_id: int = 100) -> None:
        self.chat = SimpleNamespace(id=user_id)
        self.from_user = SimpleNamespace(id=user_id)
        self.message_id = message_id
        self.bot = bot
        self.answers: list[tuple[str, Optional[Any]]] = []
        self.answer_photos: list[tuple[Any, Optional[str], Optional[Any]]] = []
        self._next_message_id = message_id + 1

    async def answer(self, text: str, reply_markup: Optional[Any] = None) -> "DummySentMessage":
        sent = DummySentMessage(self.bot, self.chat.id, self._next_message_id, text, reply_markup)
        self._next_message_id += 1
        self.answers.append((text, reply_markup))
        return sent

    async def answer_photo(
        self,
        photo: Any,
        *,
        caption: Optional[str] = None,
        reply_markup: Optional[Any] = None,
    ) -> None:
        self.answer_photos.append((photo, caption, reply_markup))


@dataclass
class DummySentMessage:
    bot: DummyBot
    chat_id: int
    message_id: int
    text: str
    reply_markup: Optional[Any]


class DummyCallback:
    def __init__(self, data: str, message: DummyMessage) -> None:
        self.data = data
        self.message = message
        self.from_user = SimpleNamespace(id=message.from_user.id)
        self._answers: list[tuple[Optional[str], bool]] = []

    async def answer(self, text: Optional[str] = None, show_alert: bool = False) -> None:
        self._answers.append((text, show_alert))


class DummyState:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.state: Optional[Any] = None

    async def update_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def set_state(self, value: Any) -> None:
        self.state = value


class StubRepository:
    def __init__(self) -> None:
        self.gender: Optional[str] = None
        self.daily_limit: int = 5
        self.daily_used: int = 0
        self.seen_models: list[str] = []
        self.updated_filters: list[tuple[int, str]] = []
        self.reminder: Optional[Any] = None
        self.synced_versions: list[str] = []

    async def ensure_user(self, user_id: int) -> Any:
        return SimpleNamespace(gender=self.gender, daily_used=self.daily_used, seen_models=list(self.seen_models))

    async def update_filters(self, user_id: int, gender: str) -> None:
        self.updated_filters.append((user_id, gender))
        self.gender = gender

    async def ensure_daily_reset(self, user_id: int) -> Any:
        return SimpleNamespace(daily_used=self.daily_used)

    async def remaining_tries(self, user_id: int) -> int:
        return max(self.daily_limit - self.daily_used, 0)

    async def record_seen_models(
        self,
        user_id: int,
        model_ids: Iterable[str],
        *,
        when: Optional[Any] = None,
        context: str = "global",
    ) -> None:
        self.seen_models.extend(model_ids)

    async def list_seen_models(self, user_id: int, *, context: str) -> set[str]:
        return set()

    async def sync_catalog_version(self, version_hash: str, *, clear_on_change: bool) -> tuple[bool, bool]:
        self.synced_versions.append(version_hash)
        return False, False

    async def inc_used_on_success(self, user_id: int) -> None:
        self.daily_used += 1

    async def set_referrer(self, user_id: int, ref_id: int) -> None:  # noqa: D401 - no-op
        return None

    async def set_reminder(self, user_id: int, when: Optional[Any]) -> None:  # noqa: D401 - no-op
        self.reminder = when

    async def list_due_reminders(self, now: Any) -> list[Any]:  # noqa: D401 - no reminders in tests
        return []


class StubRecommendationService:
    def __init__(self, models: list[GlassModel]) -> None:
        self.default = list(models)
        self.queue: list[RecommendationResult] = []
        self.calls: list[tuple[int, str]] = []

    async def recommend_for_user(
        self, user_id: int, gender: str
    ) -> RecommendationResult:
        self.calls.append((user_id, gender))
        if self.queue:
            return self.queue.pop(0)
        return RecommendationResult(models=list(self.default), exhausted=False)


class StubTryOn:
    def __init__(self, result_path: Path) -> None:
        self.result_path = result_path
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> list[Path]:
        self.calls.append(kwargs)
        return [self.result_path]


class StubStorage:
    def __init__(self, uploads_dir: Path) -> None:
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    async def allocate_upload_path(self, user_id: int, filename: str) -> Path:
        return self.uploads_dir / filename


class StubCollageBuilder:
    def __init__(self) -> None:
        self.calls: list[list[Optional[str]]] = []
        self.fail_processing = False
        self.fail_sources = False

    async def __call__(
        self, sources: Sequence[Optional[str]], cfg: CollageConfig
    ) -> io.BytesIO:
        self.calls.append(list(sources))
        if self.fail_sources:
            raise CollageSourceUnavailable("sources unavailable")
        if self.fail_processing:
            raise CollageProcessingError("processing error")
        image = Image.new("RGB", (cfg.width, cfg.height), color="white")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer


def build_router(
    tmp_path: Path,
    models: Optional[list[GlassModel]] = None,
    *,
    collage_builder: Optional[StubCollageBuilder] = None,
) -> tuple[Any, StubRepository, StubTryOn, StubCollageBuilder, StubRecommendationService]:
    repository = StubRepository()
    recommender = StubRecommendationService(models or [])
    result_path = tmp_path / "result.png"
    result_path.write_bytes(b"fake-image")
    tryon = StubTryOn(result_path=result_path)
    storage = StubStorage(tmp_path / "uploads")
    builder = collage_builder or StubCollageBuilder()
    collage_config = CollageConfig(
        width=1600,
        height=800,
        columns=2,
        margin=30,
        divider_width=6,
        divider_color="#E6E9EF",
        background="#FFFFFF",
        jpeg_quality=90,
    )

    router = setup_router(
        repository=repository,
        recommender=recommender,
        tryon=tryon,
        storage=storage,
        collage_config=collage_config,
        collage_builder=builder,
        batch_size=2,
        reminder_hours=24,
        selection_button_title_max=28,
        landing_url="https://example.com",
        promo_code="PROMO",
        no_more_message_key="all_seen",
    )
    return router, repository, tryon, builder, recommender


def get_callback_handler(router: Any, name: str):
    for handler in router.callback_query.handlers:
        if handler.callback.__name__ == name:
            return handler.callback
    raise AssertionError(f"Callback handler {name} not found")


def get_message_handler(router: Any, name: str):
    for handler in router.message.handlers:
        if handler.callback.__name__ == name:
            return handler.callback
    raise AssertionError(f"Message handler {name} not found")


def test_select_gender_deletes_prompt_and_waits_for_photo(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, repository, _, _, _ = build_router(tmp_path)
        handler = get_callback_handler(router, "select_gender")

        bot = DummyBot()
        message = DummyMessage(user_id=123, bot=bot)
        state = DummyState()
        await state.update_data(gender_prompt_message_id=message.message_id)
        callback = DummyCallback("gender_male", message)

        await handler(callback, state)

        assert state.state is TryOnStates.AWAITING_PHOTO
        assert repository.updated_filters == [(123, "male")]
        assert bot.deleted == [(123, message.message_id)]
        assert message.answers[-1][0] == msg.PHOTO_INSTRUCTION
        assert isinstance(message.answers[-1][1], ReplyKeyboardRemove)

    asyncio.run(scenario())


def test_searching_message_deleted_after_models_sent(tmp_path: Path) -> None:
    models = [
        GlassModel(
            unique_id="m1",
            title="Model 1",
            model_code="M1",
            site_url="https://example.com/1",
            img_user_url="https://example.com/1.jpg",
            img_nano_url="https://example.com/1-nano.jpg",
            gender="Мужской",
        ),
        GlassModel(
            unique_id="m2",
            title="Model 2",
            model_code="M2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="Мужской",
        ),
        GlassModel(
            unique_id="m3",
            title="Model 3",
            model_code="M3",
            site_url="https://example.com/3",
            img_user_url="https://example.com/3.jpg",
            img_nano_url="https://example.com/3-nano.jpg",
            gender="Мужской",
        ),
        GlassModel(
            unique_id="m4",
            title="Model 4",
            model_code="M4",
            site_url="https://example.com/4",
            img_user_url="https://example.com/4.jpg",
            img_nano_url="https://example.com/4-nano.jpg",
            gender="Мужской",
        ),
    ]
    async def scenario() -> None:
        router, repository, _, builder, _ = build_router(tmp_path, models=models)
        handler = get_message_handler(router, "accept_photo")

        bot = DummyBot()
        message = DummyMessage(user_id=321, bot=bot)
        message.photo = [PhotoStub("photo")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler(message, state)

        preload_id = message.message_id + 1
        assert (321, preload_id) in bot.deleted
        assert state.data.get("preload_message_id") is None
        assert len(builder.calls) == 2
        assert len(message.answer_photos) == 2
        captions = [caption for _, caption, _ in message.answer_photos]
        assert captions == [
            msg.BATCH_TITLE.format(index=1, total=2),
            msg.BATCH_TITLE.format(index=2, total=2),
        ]
        button_counts = []
        for _, _, markup in message.answer_photos:
            assert isinstance(markup, InlineKeyboardMarkup)
            assert len(markup.inline_keyboard) == 1
            button_counts.append(len(markup.inline_keyboard[0]))
        assert button_counts == [2, 2]

    asyncio.run(scenario())


def test_exhausted_message_flow(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, recommender = build_router(tmp_path)
        recommender.queue.append(RecommendationResult(models=[], exhausted=True))
        handler = get_message_handler(router, "accept_photo")

        bot = DummyBot()
        message = DummyMessage(user_id=999, bot=bot)
        message.photo = [PhotoStub("photo")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler(message, state)

        fallback_text, markup = message.answers[-1]
        assert fallback_text == msg.marketing_text("all_seen")
        assert isinstance(markup, InlineKeyboardMarkup)
        assert len(markup.inline_keyboard) == 1
        assert len(markup.inline_keyboard[0]) == 2
        assert markup.inline_keyboard[0][0].callback_data == "limit_remind"
        assert markup.inline_keyboard[0][1].url == "https://example.com"

    asyncio.run(scenario())


def test_collage_source_unavailable_falls_back_to_text(tmp_path: Path) -> None:
    models = [
        GlassModel(
            unique_id="m1",
            title="Alpha",
            model_code="A1",
            site_url="https://example.com/1",
            img_user_url="https://example.com/1.jpg",
            img_nano_url="https://example.com/1-nano.jpg",
            gender="Мужской",
        ),
        GlassModel(
            unique_id="m2",
            title="Beta",
            model_code="B2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="Мужской",
        ),
    ]

    async def scenario() -> None:
        builder = StubCollageBuilder()
        builder.fail_sources = True
        router, repository, _, returned_builder, _ = build_router(
            tmp_path, models=models, collage_builder=builder
        )
        assert returned_builder is builder
        handler = get_message_handler(router, "accept_photo")

        bot = DummyBot()
        message = DummyMessage(user_id=456, bot=bot)
        message.photo = [PhotoStub("photo")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler(message, state)

        assert len(builder.calls) == 1
        assert not message.answer_photos
        fallback_text, markup = message.answers[-1]
        assert msg.COLLAGE_IMAGES_UNAVAILABLE in fallback_text
        assert isinstance(markup, InlineKeyboardMarkup)
        assert len(markup.inline_keyboard[0]) == 2

    asyncio.run(scenario())


def test_collage_processing_error_sends_individual_photos(tmp_path: Path) -> None:
    models = [
        GlassModel(
            unique_id="m1",
            title="Alpha",
            model_code="A1",
            site_url="https://example.com/1",
            img_user_url="https://example.com/1.jpg",
            img_nano_url="https://example.com/1-nano.jpg",
            gender="Мужской",
        ),
        GlassModel(
            unique_id="m2",
            title="Beta",
            model_code="B2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="Мужской",
        ),
    ]

    async def scenario() -> None:
        builder = StubCollageBuilder()
        builder.fail_processing = True
        router, _, _, returned_builder, _ = build_router(
            tmp_path, models=models, collage_builder=builder
        )
        assert returned_builder is builder
        handler = get_message_handler(router, "accept_photo")

        bot = DummyBot()
        message = DummyMessage(user_id=789, bot=bot)
        message.photo = [PhotoStub("photo")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler(message, state)

        assert len(builder.calls) == 1
        assert len(message.answer_photos) == 2
        first_photo = message.answer_photos[0]
        second_photo = message.answer_photos[1]
        assert first_photo[1] is None
        assert first_photo[2] is None
        assert second_photo[1] == msg.BATCH_TITLE.format(index=1, total=1)
        markup = second_photo[2]
        assert isinstance(markup, InlineKeyboardMarkup)
        assert len(markup.inline_keyboard[0]) == 2

    asyncio.run(scenario())


def test_generation_message_deleted_and_caption_changes(tmp_path: Path) -> None:
    model = GlassModel(
        unique_id="m1",
        title="Model 1",
        model_code="M1",
        site_url="https://example.com/1",
        img_user_url="https://example.com/1.jpg",
        img_nano_url="https://example.com/1-nano.jpg",
        gender="Мужской",
    )
    async def scenario() -> None:
        router, repository, tryon, _, _ = build_router(tmp_path, models=[model])
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")

        bot = DummyBot()
        upload_message = DummyMessage(user_id=55, bot=bot)
        upload_message.photo = [PhotoStub("upload1")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(upload_message, state)

        data = await state.get_data()
        assert data.get("upload")

        callback = DummyCallback("pick:src=batch2:m1", upload_message)
        await handler_choose(callback, state)

        generation_id = upload_message.message_id + 2
        assert (55, generation_id) in bot.deleted
        assert state.data.get("generation_message_id") is None
        assert repository.daily_used == 1
        assert state.state is TryOnStates.RESULT
        assert upload_message.answer_photos[-1][1] == "".join(msg.FIRST_RESULT_CAPTION)
        assert tryon.calls  # ensure generation was triggered

        callback_follow = DummyCallback("more|1", upload_message)
        handler_more = get_callback_handler(router, "result_more")
        await handler_more(callback_follow, state)
        callback_second = DummyCallback("pick:src=batch2:m1", upload_message)
        await handler_choose(callback_second, state)
        assert upload_message.answer_photos[-1][1] == "".join(msg.NEXT_RESULT_CAPTION)

    asyncio.run(scenario())


def test_no_attach_photo_text_in_sources() -> None:
    project_root = Path("app")
    matches = [
        path
        for path in project_root.rglob("*.py")
        if "Прикрепить фотку" in path.read_text(encoding="utf-8")
    ]
    assert matches == []
