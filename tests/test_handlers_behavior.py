import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

from aiogram.types import InlineKeyboardMarkup
from PIL import Image

from app.config import CollageConfig
from app.fsm import ContactRequest, TryOnStates, setup_router
from app.keyboards import (
    contact_request_keyboard,
    limit_reached_keyboard,
    main_reply_keyboard,
)
from app.models import GlassModel
from app.services.collage import CollageProcessingError, CollageSourceUnavailable
from app.services.recommendation import RecommendationResult
from app.texts import messages as msg


@dataclass
class PhotoStub:
    file_unique_id: str
    file_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.file_id is None:
            self.file_id = f"{self.file_unique_id}_id"


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
        self.edited_captions: list[tuple[Optional[str], Optional[Any]]] = []
        self.edited_markups: list[Optional[Any]] = []
        self.reply_markup: Optional[Any] = None
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
    ) -> "DummySentMessage":
        sent = DummySentMessage(
            self.bot,
            self.chat.id,
            self._next_message_id,
            caption or "",
            reply_markup,
        )
        self._next_message_id += 1
        self.answer_photos.append((photo, caption, reply_markup))
        self.reply_markup = reply_markup
        return sent

    async def delete(self) -> None:
        await self.bot.delete_message(self.chat.id, self.message_id)

    async def edit_caption(
        self,
        *,
        caption: Optional[str] = None,
        reply_markup: Optional[Any] = None,
    ) -> None:
        self.edited_captions.append((caption, reply_markup))

    async def edit_reply_markup(self, reply_markup: Optional[Any] = None) -> None:
        self.edited_markups.append(reply_markup)


@dataclass
class DummySentMessage:
    bot: DummyBot
    chat_id: int
    message_id: int
    text: str
    reply_markup: Optional[Any]

    async def edit_text(
        self, text: str, reply_markup: Optional[Any] = None
    ) -> None:
        self.text = text
        self.reply_markup = reply_markup


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

    async def get_state(self) -> Optional[Any]:
        return self.state


class StubRepository:
    def __init__(self) -> None:
        self.gender: Optional[str] = None
        self.daily_limit: int = 5
        self.daily_used: int = 0
        self.seen_models: list[str] = []
        self.updated_filters: list[tuple[int, str]] = []
        self.reminder: Optional[Any] = None
        self.synced_versions: list[str] = []
        self.gen_counts: dict[int, int] = {}
        self.contact_skip: dict[int, bool] = {}
        self.contact_never: dict[int, bool] = {}
        self.contacts: dict[int, Any] = {}
        self.activity: list[int] = []

    async def ensure_user(self, user_id: int) -> Any:
        return SimpleNamespace(
            gender=self.gender,
            daily_used=self.daily_used,
            seen_models=list(self.seen_models),
            gen_count=self.gen_counts.get(user_id, 0),
            contact_skip_once=self.contact_skip.get(user_id, False),
            contact_never=self.contact_never.get(user_id, False),
        )

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

    async def touch_activity(self, user_id: int) -> None:
        self.activity.append(user_id)

    async def list_idle_reminder_candidates(self, threshold_ts: int) -> list[Any]:
        return []

    async def mark_idle_reminder_sent(self, user_id: int) -> None:
        return None

    async def get_generation_count(self, user_id: int) -> int:
        return self.gen_counts.get(user_id, 0)

    async def increment_generation_count(self, user_id: int) -> int:
        self.gen_counts[user_id] = self.gen_counts.get(user_id, 0) + 1
        return self.gen_counts[user_id]

    async def set_contact_skip_once(self, user_id: int, value: bool) -> None:
        self.contact_skip[user_id] = value

    async def set_contact_never(self, user_id: int, value: bool) -> None:
        self.contact_never[user_id] = value

    async def get_user_contact(self, user_id: int) -> Optional[Any]:
        return self.contacts.get(user_id)

    async def upsert_user_contact(self, contact: Any) -> None:
        self.contacts[contact.tg_user_id] = SimpleNamespace(
            phone_e164=contact.phone_e164,
            source=contact.source,
            consent=contact.consent,
            consent_ts=contact.consent_ts,
            reward_granted=contact.reward_granted,
        )
        self.contact_skip[contact.tg_user_id] = False
        self.contact_never[contact.tg_user_id] = False

    async def mark_contact_reward_granted(self, user_id: int) -> None:
        contact = self.contacts.get(user_id)
        if contact:
            contact.reward_granted = True

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
        self.last_user_id: Optional[int] = None

    async def generate(self, **kwargs: Any) -> list[Path]:
        self.calls.append(kwargs)
        return [self.result_path]


class StubLeadsExporter:
    def __init__(self) -> None:
        self.payloads: list[Any] = []

    async def export_lead_to_sheet(self, payload: Any) -> bool:
        self.payloads.append(payload)
        return True


class StubContactExporter:
    def __init__(self) -> None:
        self.records: list[Any] = []

    async def export_contact(self, record: Any) -> bool:
        self.records.append(record)
        return True


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
    builder = collage_builder or StubCollageBuilder()
    leads_exporter = StubLeadsExporter()
    contact_exporter = StubContactExporter()
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

    from app import fsm as fsm_module
    from app.services import drive_fetch, image_io, nanobanana

    async def fake_save_user_photo(message: DummyMessage, tmp_dir: str = "tmp") -> str:
        destination = tmp_path / f"user_{message.from_user.id}.jpg"
        destination.write_bytes(b"photo")
        tryon.last_user_id = message.from_user.id
        return str(destination)

    async def fake_redownload_user_photo(
        bot: Any, file_id: str, user_id: int, tmp_dir: str = "tmp"
    ) -> str:
        destination = tmp_path / f"user_{user_id}.jpg"
        destination.write_bytes(b"photo")
        tryon.last_user_id = user_id
        return str(destination)

    def fake_resize_inplace(path: str | Path, max_side: int = 2048) -> None:
        target = Path(path)
        if not target.exists():
            target.write_bytes(b"photo")

    glasses_path = tmp_path / "glasses.png"
    glasses_path.write_bytes(b"glasses")

    async def fake_fetch_drive_file(url: str, cache_dir: str = ".cache/frames") -> str:
        return str(glasses_path)

    async def fake_generate_glasses(
        *, face_path: str, glasses_path: str
    ) -> nanobanana.GenerationSuccess:
        tryon.calls.append(
            {
                "user_id": tryon.last_user_id,
                "input_photo": Path(face_path),
                "glasses_path": Path(glasses_path),
            }
        )
        return nanobanana.GenerationSuccess(
            image_bytes=result_path.read_bytes(),
            response={"candidates": [{"finishReason": "SUCCESS"}]},
            finish_reason="SUCCESS",
            has_inline=True,
            has_data_url=False,
            has_file_uri=False,
            attempt=1,
            retried=False,
        )

    image_io.save_user_photo = fake_save_user_photo  # type: ignore[assignment]
    image_io.redownload_user_photo = fake_redownload_user_photo  # type: ignore[assignment]
    image_io.resize_inplace = fake_resize_inplace  # type: ignore[assignment]
    drive_fetch.fetch_drive_file = fake_fetch_drive_file  # type: ignore[assignment]
    nanobanana.generate_glasses = fake_generate_glasses  # type: ignore[assignment]

    fsm_module.save_user_photo = fake_save_user_photo  # type: ignore[assignment]
    fsm_module.redownload_user_photo = fake_redownload_user_photo  # type: ignore[assignment]
    fsm_module.resize_inplace = fake_resize_inplace  # type: ignore[assignment]
    fsm_module.fetch_drive_file = fake_fetch_drive_file  # type: ignore[assignment]
    fsm_module.generate_glasses = fake_generate_glasses  # type: ignore[assignment]

    router = setup_router(
        repository=repository,
        recommender=recommender,
        collage_config=collage_config,
        collage_builder=builder,
        batch_size=2,
        reminder_hours=24,
        selection_button_title_max=28,
        site_url="https://example.com",
        promo_code="PROMO",
        no_more_message_key="all_seen",
        contact_reward_rub=1000,
        promo_contact_code="PROMO1000",
        leads_exporter=leads_exporter,
        contact_exporter=contact_exporter,
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


def assert_main_menu_keyboard(markup: Any) -> None:
    expected = main_reply_keyboard()
    assert hasattr(markup, "keyboard")
    assert markup.keyboard == expected.keyboard


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
        assert_main_menu_keyboard(message.answers[-1][1])

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
        assert captions == [None, None]
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
        assert fallback_text == msg.COLLAGE_IMAGES_UNAVAILABLE
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
        assert second_photo[1] is None
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
        initial_photo_count = len(upload_message.answer_photos)

        callback = DummyCallback("pick:src=batch2:m1", upload_message)
        await handler_choose(callback, state)

        deleted_ids = [mid for _, mid in bot.deleted]
        assert upload_message.message_id in deleted_ids
        assert any(mid > upload_message.message_id for mid in deleted_ids)
        assert state.data.get("generation_progress_message_id") is None
        assert repository.daily_used == 1
        assert state.state is TryOnStates.RESULT
        assert len(upload_message.answer_photos) == initial_photo_count + 1
        assert upload_message.answer_photos[-1][1] == "".join(msg.FIRST_RESULT_CAPTION)
        assert tryon.calls  # ensure generation was triggered

        callback_follow = DummyCallback("more|1", upload_message)
        handler_more = get_callback_handler(router, "result_more")
        await handler_more(callback_follow, state)
        assert state.state is TryOnStates.AWAITING_PHOTO
        assert upload_message.answers[-1][0] == "".join(msg.NEXT_RESULT_CAPTION)
        assert_main_menu_keyboard(upload_message.answers[-1][1])
        assert upload_message.edited_captions
        last_caption_edit = upload_message.edited_captions[-1]
        assert last_caption_edit[0] == model.title
        assert isinstance(last_caption_edit[1], InlineKeyboardMarkup)
        assert [
            button.text for button in last_caption_edit[1].inline_keyboard[0]
        ] == [msg.DETAILS_BUTTON_TEXT]

        upload_message.photo = [PhotoStub("upload2")]  # type: ignore[attr-defined]
        await handler_photo(upload_message, state)

        callback_second = DummyCallback("pick:src=batch2:m1", upload_message)
        await handler_choose(callback_second, state)
        second_result = upload_message.answer_photos[-1]
        assert second_result[1] == model.title
        assert isinstance(second_result[2], InlineKeyboardMarkup)
        assert [
            button.text for button in second_result[2].inline_keyboard[0]
        ] == [msg.DETAILS_BUTTON_TEXT]
        assert state.state is ContactRequest.waiting_for_phone
        assert upload_message.answers[-1][0].startswith(
            f"<b>{msg.ASK_PHONE_TITLE}"
        )

    asyncio.run(scenario())


def test_generation_unsuitable_photo_requests_new_upload(tmp_path: Path) -> None:
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
        from app import fsm as fsm_module
        from app.services import nanobanana

        router, _, _, _, _ = build_router(tmp_path, models=[model])
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")

        bot = DummyBot()
        message = DummyMessage(user_id=72, bot=bot)
        message.photo = [PhotoStub("upload1")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(message, state)

        failing_error = nanobanana.NanoBananaGenerationError(
            "unsuitable",
            reason_code="UNSUITABLE_PHOTO",
            reason_detail="finish=SAFETY",
            finish_reason="SAFETY",
            has_inline=False,
            has_data_url=False,
            has_file_uri=False,
        )

        async def failing_generate_glasses(*, face_path: str, glasses_path: str):
            raise failing_error

        original_generate = nanobanana.generate_glasses
        original_fsm_generate = fsm_module.generate_glasses
        nanobanana.generate_glasses = failing_generate_glasses  # type: ignore[assignment]
        fsm_module.generate_glasses = failing_generate_glasses  # type: ignore[assignment]
        try:
            callback = DummyCallback("pick:src=batch2:m1", message)
            await handler_choose(callback, state)
        finally:
            nanobanana.generate_glasses = original_generate  # type: ignore[assignment]
            fsm_module.generate_glasses = original_fsm_generate  # type: ignore[assignment]

        assert state.state is TryOnStates.AWAITING_PHOTO
        assert state.data.get("selected_model") is None
        assert state.data.get("upload") is None
        assert state.data.get("upload_file_id") is None
        assert state.data.get("current_models") == []
        assert message.answers[-1][0] == msg.PHOTO_NOT_SUITABLE_MAIN
        markup = message.answers[-1][1]
        assert isinstance(markup, InlineKeyboardMarkup)
        assert markup.inline_keyboard[0][0].text == msg.BTN_SEND_NEW_PHOTO

    asyncio.run(scenario())


def test_generation_transient_error_requests_new_upload(tmp_path: Path) -> None:
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
        from app import fsm as fsm_module
        from app.services import nanobanana

        router, _, _, _, _ = build_router(tmp_path, models=[model])
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")

        bot = DummyBot()
        message = DummyMessage(user_id=73, bot=bot)
        message.photo = [PhotoStub("upload1")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(message, state)

        transient_error = nanobanana.NanoBananaGenerationError(
            "transient",
            reason_code="TRANSIENT",
            reason_detail="timeout",
            finish_reason="OTHER",
            has_inline=False,
            has_data_url=False,
            has_file_uri=False,
        )

        async def transient_generate_glasses(*, face_path: str, glasses_path: str):
            raise transient_error

        original_generate = nanobanana.generate_glasses
        original_fsm_generate = fsm_module.generate_glasses
        nanobanana.generate_glasses = transient_generate_glasses  # type: ignore[assignment]
        fsm_module.generate_glasses = transient_generate_glasses  # type: ignore[assignment]
        try:
            callback = DummyCallback("pick:src=batch2:m1", message)
            await handler_choose(callback, state)
        finally:
            nanobanana.generate_glasses = original_generate  # type: ignore[assignment]
            fsm_module.generate_glasses = original_fsm_generate  # type: ignore[assignment]

        assert state.state is TryOnStates.AWAITING_PHOTO
        assert state.data.get("selected_model") is None
        assert state.data.get("upload") is None
        assert state.data.get("upload_file_id") is None
        assert state.data.get("current_models") == []
        assert message.answers[-1][0] == msg.PHOTO_NOT_SUITABLE_MAIN
        markup = message.answers[-1][1]
        assert isinstance(markup, InlineKeyboardMarkup)
        assert markup.inline_keyboard[0][0].text == msg.BTN_SEND_NEW_PHOTO

    asyncio.run(scenario())


def test_idle_reminder_message_removed_when_user_requests_more(tmp_path: Path) -> None:
    models = [
        GlassModel(
            unique_id="m1",
            title="Model 1",
            model_code="M1",
            site_url="https://example.com/1",
            img_user_url="https://example.com/1.jpg",
            img_nano_url="https://example.com/1-nano.jpg",
            gender="Мужской",
        )
    ]

    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path, models=models)
        handler_more = get_callback_handler(router, "result_more")

        bot = DummyBot()
        message = DummyMessage(user_id=555, bot=bot)
        state = DummyState()
        await state.update_data(gender="male")
        await state.set_state(TryOnStates.RESULT)

        callback = DummyCallback("more|idle", message)
        await handler_more(callback, state)

        assert message.answers[-1][0] == msg.SEARCHING_MODELS_PROMPT
        assert (555, message.message_id) in bot.deleted

    asyncio.run(scenario())


def test_contact_prompt_after_second_generation(tmp_path: Path) -> None:
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
    ]

    async def scenario() -> None:
        router, repository, _, _, _ = build_router(tmp_path, models=models)
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")
        handler_more = get_callback_handler(router, "result_more")

        bot = DummyBot()
        message = DummyMessage(user_id=777, bot=bot)
        message.photo = [PhotoStub("photo1")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(message, state)

        first_callback = DummyCallback("pick:src=batch2:m1", message)
        await handler_choose(first_callback, state)
        assert repository.gen_counts[777] == 1
        assert not any(msg.ASK_PHONE_TITLE in text for text, _ in message.answers)

        more_callback = DummyCallback("more|1", message)
        await handler_more(more_callback, state)
        assert state.state is TryOnStates.AWAITING_PHOTO
        assert message.answers[-1][0] == "".join(msg.NEXT_RESULT_CAPTION)
        assert_main_menu_keyboard(message.answers[-1][1])

        message.photo = [PhotoStub("photo2")]  # type: ignore[attr-defined]
        await handler_photo(message, state)
        assert repository.gen_counts[777] == 1

        second_callback = DummyCallback("pick:src=batch2:m1", message)
        await handler_choose(second_callback, state)
        assert repository.gen_counts[777] == 2

        second_result = message.answer_photos[-1]
        assert second_result[1] == models[0].title
        assert isinstance(second_result[2], InlineKeyboardMarkup)
        assert [
            button.text for button in second_result[2].inline_keyboard[0]
        ] == [msg.DETAILS_BUTTON_TEXT]
        assert state.state is ContactRequest.waiting_for_phone
        assert message.answers[-1][0].startswith(f"<b>{msg.ASK_PHONE_TITLE}")
        assert (
            message.answers[-1][1].keyboard
            == contact_request_keyboard().keyboard
        )

    asyncio.run(scenario())


def test_contact_share_sends_followup_without_new_selection(tmp_path: Path) -> None:
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
    ]

    async def scenario() -> None:
        router, repository, _, _, _ = build_router(tmp_path, models=models)
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")
        handler_more = get_callback_handler(router, "result_more")
        contact_handler = get_message_handler(router, "contact_shared")

        bot = DummyBot()
        message = DummyMessage(user_id=888, bot=bot)
        message.photo = [PhotoStub("photo1")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(message, state)
        await handler_choose(DummyCallback("pick:src=batch2:m1", message), state)
        await handler_more(DummyCallback("more|1", message), state)
        assert state.state is TryOnStates.AWAITING_PHOTO

        message.photo = [PhotoStub("photo2")]  # type: ignore[attr-defined]
        await handler_photo(message, state)
        await handler_choose(DummyCallback("pick:src=batch2:m1", message), state)

        assert state.state is ContactRequest.waiting_for_phone
        photos_before = list(message.answer_photos)

        contact_message = DummyMessage(user_id=888, bot=bot, message_id=400)
        contact_message.contact = SimpleNamespace(phone_number="+79991234567")

        await contact_handler(contact_message, state)

        assert state.state is TryOnStates.AWAITING_PHOTO
        assert list(message.answer_photos) == photos_before
        assert contact_message.answers[0][0] == msg.ASK_PHONE_THANKS.format(
            rub=1000, promo="PROMO1000"
        )
        assert_main_menu_keyboard(contact_message.answers[0][1])
        assert contact_message.answers[1][0] == "".join(msg.NEXT_RESULT_CAPTION)
        assert_main_menu_keyboard(contact_message.answers[1][1])

    asyncio.run(scenario())


def test_limit_result_sends_card_and_summary_message(tmp_path: Path) -> None:
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
        router, repository, _, _, _ = build_router(tmp_path, models=[model])
        repository.daily_limit = 1
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")

        bot = DummyBot()
        message = DummyMessage(user_id=999, bot=bot)
        message.photo = [PhotoStub("limit-photo")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(message, state)
        await handler_choose(DummyCallback("pick:src=batch2:m1", message), state)

        result_photo = message.answer_photos[-1]
        assert result_photo[1] == model.title
        assert isinstance(result_photo[2], InlineKeyboardMarkup)
        assert [
            button.text for button in result_photo[2].inline_keyboard[0]
        ] == [msg.DETAILS_BUTTON_TEXT]

        limit_message = message.answers[-1]
        assert limit_message[0] == "".join(msg.DAILY_LIMIT_MESSAGE)
        expected_markup = limit_reached_keyboard("https://example.com")
        assert isinstance(limit_message[1], InlineKeyboardMarkup)
        assert (
            limit_message[1].inline_keyboard
            == expected_markup.inline_keyboard
        )
        assert state.state is TryOnStates.DAILY_LIMIT_REACHED

    asyncio.run(scenario())


def test_no_attach_photo_text_in_sources() -> None:
    project_root = Path("app")
    matches = [
        path
        for path in project_root.rglob("*.py")
        if "Прикрепить фотку" in path.read_text(encoding="utf-8")
    ]
    assert matches == []
