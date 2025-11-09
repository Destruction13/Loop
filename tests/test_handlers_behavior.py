import asyncio
import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardRemove
from PIL import Image

from app.config import CollageConfig
from app.fsm import ContactRequest, TryOnStates, setup_router
from app.keyboards import (
    REUSE_SAME_PHOTO_CALLBACK,
    contact_request_keyboard,
    contact_share_reply_keyboard,
    limit_reached_keyboard,
    promo_keyboard,
    reuse_same_photo_keyboard,
)
from app.models import GlassModel
from app.services.collage import CollageProcessingError, CollageSourceUnavailable
from app.services.recommendation import RecommendationResult
from app.texts import messages as msg


TEST_PRIVACY_POLICY_URL = "https://example.com/privacy"


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
        self.chat_actions: list[tuple[int, Any]] = []
        self.edited_markups: list[tuple[int, int, Any]] = []
        self.edited_texts: list[tuple[int, int, str, Optional[Any]]] = []

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        self.deleted.append((chat_id, message_id))

    async def download(self, photo: PhotoStub, destination: Path) -> None:
        self.downloads.append((photo, destination))

    async def send_chat_action(self, chat_id: int, action: Any) -> None:
        self.chat_actions.append((chat_id, action))

    async def edit_message_reply_markup(
        self,
        *,
        chat_id: int,
        message_id: int,
        reply_markup: Optional[Any] = None,
    ) -> None:
        self.edited_markups.append((chat_id, message_id, reply_markup))

    async def edit_message_text(
        self,
        text: str,
        *,
        chat_id: int,
        message_id: int,
        reply_markup: Optional[Any] = None,
    ) -> None:
        self.edited_texts.append((chat_id, message_id, text, reply_markup))


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

    async def answer(
        self, text: str, reply_markup: Optional[Any] = None, **_: Any
    ) -> "DummySentMessage":
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
        **_: Any,
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
        if caption:
            clean_caption = (
                caption.replace("<b>", "")
                .replace("</b>", "")
                .replace("<i>", "")
                .replace("</i>", "")
            )
            title_line = clean_caption.split("\n", 1)[0]
            self.edited_captions.append((title_line, None))
        return sent

    async def delete(self) -> None:
        await self.bot.delete_message(self.chat.id, self.message_id)

    async def edit_caption(
        self,
        *,
        caption: Optional[str] = None,
        reply_markup: Optional[Any] = None,
        **_: Any,
    ) -> None:
        clean_caption = caption
        if clean_caption is not None:
            clean_caption = (
                clean_caption.replace("<b>", "")
                .replace("</b>", "")
                .replace("<i>", "")
                .replace("</i>", "")
            )
        self.edited_captions.append((clean_caption, reply_markup))

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
        self.bot = message.bot

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

    async def clear(self) -> None:
        self.data.clear()
        self.state = None


class StubRepository:
    def __init__(self) -> None:
        self.gender: Optional[str] = None
        self.daily_limit: int = 5
        self.tries_used: int = 0
        self.seen_models: list[str] = []
        self.updated_filters: list[tuple[int, str]] = []
        self.reminder: Optional[Any] = None
        self.synced_versions: list[str] = []
        self.gen_counts: dict[int, int] = {}
        self.contact_skip: dict[int, bool] = {}
        self.contact_never: dict[int, bool] = {}
        self.contacts: dict[int, Any] = {}
        self.contact_generations_today: dict[int, int] = {}
        self.contact_cycle_started_at: dict[int, datetime] = {}
        self.contact_prompt_second: set[int] = set()
        self.contact_prompt_sixth: set[int] = set()
        self.activity: list[int] = []
        self.locked_until: Optional[datetime] = None
        self.cycle_index: int = 0
        self.nudge_sent: bool = False
        self.more_buttons: dict[int, tuple[Optional[int], Optional[str], Optional[dict[str, Any]]]] = {}
        self.saved_contacts: list[tuple[int, str]] = []

    async def ensure_user(self, user_id: int) -> Any:
        message_id, message_type, payload = self.more_buttons.get(
            user_id, (None, None, None)
        )
        return SimpleNamespace(
            gender=self.gender,
            tries_used=self.tries_used,
            daily_try_limit=self.daily_limit,
            locked_until=self.locked_until,
            nudge_sent_cycle=self.nudge_sent,
            cycle_index=self.cycle_index,
            seen_models=list(self.seen_models),
            gen_count=self.gen_counts.get(user_id, 0),
            contact_skip_once=self.contact_skip.get(user_id, False),
            contact_never=self.contact_never.get(user_id, False),
            contact_generations_today=self.contact_generations_today.get(user_id, 0),
            contact_generations_started_at=self.contact_cycle_started_at.get(user_id),
            contact_prompt_second_sent=user_id in self.contact_prompt_second,
            contact_prompt_sixth_sent=user_id in self.contact_prompt_sixth,
            last_more_message_id=message_id,
            last_more_message_type=message_type,
            last_more_message_payload=payload,
        )

    async def update_filters(self, user_id: int, gender: str) -> None:
        self.updated_filters.append((user_id, gender))
        self.gender = gender

    async def ensure_daily_reset(self, user_id: int, *, now: Optional[datetime] = None) -> Any:
        if self.locked_until and now and now >= self.locked_until:
            self.locked_until = None
            self.tries_used = 0
            self.cycle_index += 1
            self.nudge_sent = False
        return SimpleNamespace(
            tries_used=self.tries_used,
            daily_try_limit=self.daily_limit,
            locked_until=self.locked_until,
            nudge_sent_cycle=self.nudge_sent,
        )

    async def remaining_tries(self, user_id: int) -> int:
        if self.locked_until is not None:
            return 0
        return max(self.daily_limit - self.tries_used, 0)

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
        if self.locked_until is not None:
            return
        self.tries_used += 1
        if self.tries_used >= self.daily_limit:
            self.locked_until = datetime.now(UTC) + timedelta(hours=24)

    async def touch_activity(self, user_id: int) -> None:
        self.activity.append(user_id)

    async def list_idle_reminder_candidates(self, threshold_ts: int) -> list[Any]:
        return []

    async def mark_idle_reminder_sent(self, user_id: int) -> None:
        return None

    async def mark_cycle_nudge_sent(self, user_id: int) -> None:
        self.nudge_sent = True

    async def set_last_more_message(
        self,
        user_id: int,
        message_id: Optional[int],
        message_type: Optional[str],
        payload: Optional[dict[str, Any]],
    ) -> None:
        self.more_buttons[user_id] = (message_id, message_type, payload)

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

    async def save_contact(self, user_id: int, phone_number: str) -> None:
        self.saved_contacts.append((user_id, phone_number))

    async def register_contact_generation(
        self,
        user_id: int,
        *,
        now: Optional[datetime] = None,
        initial_trigger: int,
        reminder_trigger: int,
    ) -> tuple[int, Optional[str]]:
        moment = now or datetime.now(tz=UTC)
        started = self.contact_cycle_started_at.get(user_id)
        if started is None or moment - started >= timedelta(hours=24):
            self.contact_cycle_started_at[user_id] = moment
            self.contact_generations_today[user_id] = 0
            self.contact_prompt_second.discard(user_id)
            self.contact_prompt_sixth.discard(user_id)
            self.contact_skip[user_id] = False
        count = self.contact_generations_today.get(user_id, 0) + 1
        self.contact_generations_today[user_id] = count
        trigger: Optional[str] = None
        if count == initial_trigger and user_id not in self.contact_prompt_second:
            trigger = "second"
        elif count == reminder_trigger and user_id not in self.contact_prompt_sixth:
            trigger = "sixth"
        return count, trigger

    async def mark_contact_prompt_sent(self, user_id: int, trigger: str) -> None:
        if trigger == "second":
            self.contact_prompt_second.add(user_id)
        elif trigger == "sixth":
            self.contact_prompt_sixth.add(user_id)

    async def set_referrer(self, user_id: int, ref_id: int) -> None:  # noqa: D401 - no-op
        return None

    async def set_reminder(self, user_id: int, when: Optional[Any]) -> None:  # noqa: D401 - no-op
        self.reminder = when

    async def list_due_reminders(self, now: Any) -> list[Any]:  # noqa: D401 - no reminders in tests
        return []

    async def reset_user_session(self, user_id: int) -> None:  # noqa: D401 - no-op
        return None


class StubRecommendationService:
    def __init__(self, models: list[GlassModel]) -> None:
        self.default = list(models)
        self.queue: list[RecommendationResult] = []
        self.calls: list[tuple[int, str]] = []

    async def recommend_for_user(
        self,
        user_id: int,
        gender: str,
        *,
        exclude_ids: set[str] | None = None,
    ) -> RecommendationResult:
        self.calls.append((user_id, gender))
        exclude = set(exclude_ids or set())
        if self.queue:
            result = self.queue.pop(0)
        else:
            result = RecommendationResult(models=list(self.default), exhausted=False)
        filtered_models = [
            model for model in result.models if model.unique_id not in exclude
        ]
        exhausted = result.exhausted or not filtered_models
        return RecommendationResult(models=filtered_models, exhausted=exhausted)


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
        width = cfg.slot_width * 2 + cfg.separator_width + cfg.padding * 2
        height = cfg.slot_height + cfg.padding * 2
        fmt = (cfg.output_format or "PNG").upper()
        mode = "RGB" if fmt == "JPEG" else "RGBA"
        image = Image.new(mode, (width, height), color=(255, 255, 255, 255))
        buffer = io.BytesIO()
        save_kwargs: dict[str, object] = {"format": fmt}
        if fmt == "JPEG":
            save_kwargs["quality"] = cfg.jpeg_quality
        image.save(buffer, **save_kwargs)
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
        slot_width=1080,
        slot_height=1440,
        separator_width=24,
        padding=48,
        separator_color="#E6E9EF",
        background="#FFFFFF",
        output_format="PNG",
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

    async def fake_track_event(*args: Any, **kwargs: Any) -> None:
        return None

    fsm_module.track_event = fake_track_event  # type: ignore[assignment]

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
        idle_nudge_seconds=0,
        enable_idle_nudge=False,
        privacy_policy_url=TEST_PRIVACY_POLICY_URL,
        promo_video_path=tmp_path / "promo.mp4",
        promo_video_enabled=False,
        promo_video_width=None,
        promo_video_height=None,
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


def assert_no_reply_keyboard(markup: Any) -> None:
    assert markup is None


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
        assert_no_reply_keyboard(message.answers[-1][1])

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
            gender="male",
        ),
        GlassModel(
            unique_id="m2",
            title="Model 2",
            model_code="M2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="male",
        ),
        GlassModel(
            unique_id="m3",
            title="Model 3",
            model_code="M3",
            site_url="https://example.com/3",
            img_user_url="https://example.com/3.jpg",
            img_nano_url="https://example.com/3-nano.jpg",
            gender="male",
        ),
        GlassModel(
            unique_id="m4",
            title="Model 4",
            model_code="M4",
            site_url="https://example.com/4",
            img_user_url="https://example.com/4.jpg",
            img_nano_url="https://example.com/4-nano.jpg",
            gender="male",
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


def test_small_catalog_keeps_showing_models(tmp_path: Path) -> None:
    model = GlassModel(
        unique_id="only-one",
        title="Solo",
        model_code="S1",
        site_url="https://example.com/solo",
        img_user_url="https://example.com/solo.jpg",
        img_nano_url="https://example.com/solo-nano.jpg",
        gender="male",
    )

    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path, models=[model])
        handler = get_message_handler(router, "accept_photo")

        bot = DummyBot()
        message = DummyMessage(user_id=4242, bot=bot)
        message.photo = [PhotoStub("photo")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler(message, state)

        assert message.answer_photos, "Expected at least one recommendation photo"
        texts = [text for text, _ in message.answers]
        assert msg.marketing_text("all_seen") not in texts

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
        assert fallback_text == msg.CATALOG_TEMPORARILY_UNAVAILABLE
        assert markup is None

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
            gender="male",
        ),
        GlassModel(
            unique_id="m2",
            title="Beta",
            model_code="B2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="male",
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
            gender="male",
        ),
        GlassModel(
            unique_id="m2",
            title="Beta",
            model_code="B2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="male",
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
        gender="male",
    )
    async def scenario() -> None:
        alt_model = GlassModel(
            unique_id="m2",
            title="Model 2",
            model_code="M2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="male",
        )
        router, repository, tryon, _, recommender = build_router(
            tmp_path, models=[model]
        )
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

        assert upload_message.edited_captions
        first_edit_caption, first_edit_markup = upload_message.edited_captions[0]
        assert first_edit_caption == model.title
        assert first_edit_markup is None
        assert state.data.get("generation_progress_message_id") is None
        assert repository.tries_used == 1
        assert state.state is TryOnStates.RESULT
        assert len(upload_message.answer_photos) == initial_photo_count + 1
        expected_first = "".join(msg.FIRST_RESULT_CAPTION).strip()
        assert (
            upload_message.answer_photos[-1][1]
            == f"<b>{model.title}</b>\n\n{expected_first}"
        )
        assert tryon.calls  # ensure generation was triggered

        callback_follow = DummyCallback("more|1", upload_message)
        handler_more = get_callback_handler(router, "result_more")
        recommender.queue.append(
            RecommendationResult(models=[alt_model], exhausted=False)
        )
        previous_photo_messages = len(upload_message.answer_photos)
        await handler_more(callback_follow, state)
        assert state.state is TryOnStates.SHOW_RECS
        assert upload_message.answers[-1][0] == msg.SEARCHING_MODELS_PROMPT
        assert len(upload_message.answer_photos) == previous_photo_messages + 1
        followup_photo = upload_message.answer_photos[-1]
        assert isinstance(followup_photo[2], InlineKeyboardMarkup)
        assert upload_message.edited_captions
        last_caption_edit = upload_message.edited_captions[-1]
        assert last_caption_edit[0] == model.title
        assert isinstance(last_caption_edit[1], InlineKeyboardMarkup)
        assert [
            button.text for button in last_caption_edit[1].inline_keyboard[0]
        ] == [msg.DETAILS_BUTTON_TEXT]

        callback_second = DummyCallback(
            f"pick:src=batch2:{alt_model.unique_id}", upload_message
        )
        await handler_choose(callback_second, state)
        second_result = upload_message.answer_photos[-1]
        expected_second = "".join(msg.SECOND_RESULT_CAPTION).strip()
        assert second_result[1] == f"<b>{alt_model.title}</b>\n\n{expected_second}"
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
        gender="male",
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
        gender="male",
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
            gender="male",
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

        assert state.state is TryOnStates.AWAITING_PHOTO
        assert message.answers[-1][0] == msg.PHOTO_INSTRUCTION
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
            gender="male",
        ),
        GlassModel(
            unique_id="m2",
            title="Model 2",
            model_code="M2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="male",
        ),
    ]

    async def scenario() -> None:
        extra_model = GlassModel(
            unique_id="m3",
            title="Model 3",
            model_code="M3",
            site_url="https://example.com/3",
            img_user_url="https://example.com/3.jpg",
            img_nano_url="https://example.com/3-nano.jpg",
            gender="male",
        )
        router, repository, _, _, recommender = build_router(
            tmp_path, models=models
        )
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
        recommender.queue.append(
            RecommendationResult(models=[extra_model], exhausted=False)
        )
        previous_photos = list(message.answer_photos)
        await handler_more(more_callback, state)
        assert state.state is TryOnStates.SHOW_RECS
        assert message.answers[-1][0] == msg.SEARCHING_MODELS_PROMPT
        assert len(message.answer_photos) == len(previous_photos) + 1

        second_callback = DummyCallback(
            f"pick:src=batch2:{extra_model.unique_id}", message
        )
        await handler_choose(second_callback, state)
        assert repository.gen_counts[777] == 2

        second_result = message.answer_photos[-1]
        expected_second = "".join(msg.SECOND_RESULT_CAPTION).strip()
        assert second_result[1] == f"<b>{extra_model.title}</b>\n\n{expected_second}"
        assert isinstance(second_result[2], InlineKeyboardMarkup)
        assert [
            button.text for button in second_result[2].inline_keyboard[0]
        ] == [msg.DETAILS_BUTTON_TEXT]
        assert state.state is ContactRequest.waiting_for_phone
        assert message.answers[-1][0].startswith(f"<b>{msg.ASK_PHONE_TITLE}")
        prompt_markup = message.answers[-1][1]
        assert isinstance(prompt_markup, InlineKeyboardMarkup)
        assert (
            prompt_markup.inline_keyboard
            == contact_request_keyboard().inline_keyboard
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
            gender="male",
        ),
        GlassModel(
            unique_id="m2",
            title="Model 2",
            model_code="M2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="male",
        ),
    ]

    async def scenario() -> None:
        extra_model = GlassModel(
            unique_id="m3",
            title="Model 3",
            model_code="M3",
            site_url="https://example.com/3",
            img_user_url="https://example.com/3.jpg",
            img_nano_url="https://example.com/3-nano.jpg",
            gender="male",
        )
        router, repository, _, _, recommender = build_router(
            tmp_path, models=models
        )
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
        recommender.queue.append(
            RecommendationResult(models=[extra_model], exhausted=False)
        )
        await handler_more(DummyCallback("more|1", message), state)
        assert state.state is TryOnStates.SHOW_RECS

        await handler_choose(
            DummyCallback(f"pick:src=batch2:{extra_model.unique_id}", message), state
        )

        assert state.state is ContactRequest.waiting_for_phone
        photos_before = list(message.answer_photos)

        contact_message = DummyMessage(user_id=888, bot=bot, message_id=400)
        contact_message.contact = SimpleNamespace(phone_number="+79991234567")

        await contact_handler(contact_message, state)

        assert state.state is TryOnStates.RESULT
        assert repository.saved_contacts == [(888, "+79991234567")]
        assert list(message.answer_photos) == photos_before
        assert len(contact_message.answers) == 2
        thanks_text, thanks_markup = contact_message.answers[0]
        assert thanks_text == msg.ASK_PHONE_THANKS.format(
            rub=1000, promo="PROMO1000"
        )
        assert isinstance(thanks_markup, ReplyKeyboardRemove)
        followup_text, followup_markup = contact_message.answers[1]
        assert followup_text == "".join(msg.THIRD_RESULT_CAPTION)
        assert isinstance(followup_markup, InlineKeyboardMarkup)
        assert (
            followup_markup.inline_keyboard
            == reuse_same_photo_keyboard().inline_keyboard
        )
        state_data = await state.get_data()
        assert state_data.get("suppress_more_button") is True
        assert state_data.get("contact_prompt_message_id") is None

    asyncio.run(scenario())


def test_contact_share_button_requests_contact_keyboard(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path)
        handler = get_callback_handler(router, "contact_share_button")

        bot = DummyBot()
        message = DummyMessage(user_id=321, bot=bot)
        state = DummyState()
        await state.set_state(ContactRequest.waiting_for_phone.state)
        await state.update_data(
            contact_prompt_message_id=message.message_id,
            last_aux_message_id=message.message_id,
        )

        callback = DummyCallback("contact_share", message)

        await handler(callback, state)

        assert bot.deleted == [(321, message.message_id)]
        assert message.answers[-1][0] == msg.ASK_PHONE_PROMPT_MANUAL
        markup = message.answers[-1][1]
        expected = contact_share_reply_keyboard()
        assert type(markup) is type(expected)
        assert markup.keyboard == expected.keyboard
        assert markup.one_time_keyboard == expected.one_time_keyboard

    asyncio.run(scenario())


def test_reuse_same_photo_suppresses_more_button(tmp_path: Path) -> None:
    model = GlassModel(
        unique_id="m1",
        title="Model 1",
        model_code="M1",
        site_url="https://example.com/1",
        img_user_url="https://example.com/1.jpg",
        img_nano_url="https://example.com/1-nano.jpg",
        gender="male",
    )
    new_model = GlassModel(
        unique_id="m2",
        title="Model 2",
        model_code="M2",
        site_url="https://example.com/2",
        img_user_url="https://example.com/2.jpg",
        img_nano_url="https://example.com/2-nano.jpg",
        gender="male",
    )

    async def scenario() -> None:
        router, repository, tryon, _, recommender = build_router(
            tmp_path, models=[model]
        )
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")
        reuse_handler = get_callback_handler(router, "reuse_same_photo")

        bot = DummyBot()
        message = DummyMessage(user_id=654, bot=bot)
        message.photo = [PhotoStub("photo1")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(message, state)
        await handler_choose(DummyCallback("pick:src=batch2:m1", message), state)

        assert repository.gen_counts[654] == 1
        assert state.state is TryOnStates.RESULT

        recommender.queue.append(
            RecommendationResult(models=[new_model], exhausted=False)
        )
        await state.update_data(suppress_more_button=True)

        previous_photos = list(message.answer_photos)

        callback = DummyCallback(REUSE_SAME_PHOTO_CALLBACK, message)
        await reuse_handler(callback, state)

        assert state.state is TryOnStates.SHOW_RECS
        assert message.answers[-1][0] == msg.SEARCHING_MODELS_PROMPT
        assert len(message.answer_photos) == len(previous_photos) + 1
        collage_photo = message.answer_photos[-1]
        assert collage_photo[1] is None
        assert isinstance(collage_photo[2], InlineKeyboardMarkup)
        option_texts = [button.text for button in collage_photo[2].inline_keyboard[0]]
        assert option_texts

        await handler_choose(
            DummyCallback(f"pick:src=batch2:{new_model.unique_id}", message), state
        )

        result_photo = message.answer_photos[-1]
        expected_caption = "".join(msg.SECOND_RESULT_CAPTION).strip()
        assert result_photo[1] == f"<b>{new_model.title}</b>\n\n{expected_caption}"
        result_markup = result_photo[2]
        assert isinstance(result_markup, InlineKeyboardMarkup)
        assert len(result_markup.inline_keyboard) == 1
        assert [button.text for button in result_markup.inline_keyboard[0]] == [
            msg.DETAILS_BUTTON_TEXT
        ]
        assert repository.gen_counts[654] == 2
        assert len(tryon.calls) >= 2
        assert state.state == ContactRequest.waiting_for_phone.state
        data_after = await state.get_data()
        assert data_after.get("suppress_more_button") is True

    asyncio.run(scenario())


def test_contact_prompt_after_skip_returns_next_cycle(tmp_path: Path) -> None:
    models = [
        GlassModel(
            unique_id="m1",
            title="Model 1",
            model_code="M1",
            site_url="https://example.com/1",
            img_user_url="https://example.com/1.jpg",
            img_nano_url="https://example.com/1-nano.jpg",
            gender="male",
        ),
        GlassModel(
            unique_id="m2",
            title="Model 2",
            model_code="M2",
            site_url="https://example.com/2",
            img_user_url="https://example.com/2.jpg",
            img_nano_url="https://example.com/2-nano.jpg",
            gender="male",
        ),
    ]

    async def scenario() -> None:
        router, repository, _, _, _ = build_router(tmp_path, models=models)
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")

        user_id = 889
        repository.gen_counts[user_id] = 2
        repository.contact_skip[user_id] = True
        repository.contact_generations_today[user_id] = 2
        repository.contact_cycle_started_at[user_id] = datetime.now(UTC)
        repository.contact_prompt_second.add(user_id)

        bot = DummyBot()
        message = DummyMessage(user_id=user_id, bot=bot)
        message.photo = [PhotoStub("photo1")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=False)

        answers_before = len(message.answers)
        await handler_photo(message, state)
        await handler_choose(DummyCallback("pick:src=batch2:m1", message), state)

        assert state.state is TryOnStates.RESULT
        recent_answers = message.answers[answers_before:]
        assert not any(msg.ASK_PHONE_TITLE in text for text, _ in recent_answers)

        repository.contact_cycle_started_at[user_id] = datetime.now(UTC) - timedelta(
            hours=25
        )
        message.photo = [PhotoStub("photo2")]  # type: ignore[attr-defined]
        answers_before = len(message.answers)
        await handler_photo(message, state)
        await handler_choose(DummyCallback("pick:src=batch2:m1", message), state)

        assert state.state is TryOnStates.RESULT
        second_wave = message.answers[answers_before:]
        assert not any(msg.ASK_PHONE_TITLE in text for text, _ in second_wave)

        message.photo = [PhotoStub("photo3")]  # type: ignore[attr-defined]
        await handler_photo(message, state)
        callback = DummyCallback("pick:src=batch2:m1", message)
        await handler_choose(callback, state)

        assert repository.gen_counts[user_id] >= 5
        assert len(message.answer_photos) >= 1
        assert state.state is ContactRequest.waiting_for_phone
        assert message.answers[-1][0].startswith(f"<b>{msg.ASK_PHONE_TITLE}")

    asyncio.run(scenario())


def test_limit_result_sends_card_and_summary_message(tmp_path: Path) -> None:
    model = GlassModel(
        unique_id="m1",
        title="Model 1",
        model_code="M1",
        site_url="https://example.com/1",
        img_user_url="https://example.com/1.jpg",
        img_nano_url="https://example.com/1-nano.jpg",
        gender="male",
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
        assert result_photo[1] == f"<b>{model.title}</b>"
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


def test_limit_promo_removes_limit_message(tmp_path: Path) -> None:
    model = GlassModel(
        unique_id="m1",
        title="Model 1",
        model_code="M1",
        site_url="https://example.com/1",
        img_user_url="https://example.com/1.jpg",
        img_nano_url="https://example.com/1-nano.jpg",
        gender="male",
    )

    async def scenario() -> None:
        router, repository, _, _, _ = build_router(tmp_path, models=[model])
        repository.daily_limit = 1
        handler_photo = get_message_handler(router, "accept_photo")
        handler_choose = get_callback_handler(router, "choose_model")
        handler_limit_promo = get_callback_handler(router, "limit_promo")

        bot = DummyBot()
        message = DummyMessage(user_id=111, bot=bot)
        message.photo = [PhotoStub("limit-photo")]  # type: ignore[attr-defined]
        state = DummyState()
        await state.update_data(gender="male", first_generated_today=True)

        await handler_photo(message, state)
        await handler_choose(DummyCallback("pick:src=batch2:m1", message), state)

        limit_message_id = state.data.get("last_aux_message_id")
        assert isinstance(limit_message_id, int)

        callback = DummyCallback("limit_promo", message)
        await handler_limit_promo(callback, state)

        assert (message.chat.id, limit_message_id) in bot.deleted

        promo_message = message.answers[-1]
        expected_text = msg.PROMO_MESSAGE_TEMPLATE.format(promo_code="PROMO")
        assert promo_message[0] == expected_text
        expected_markup = promo_keyboard("https://example.com")
        assert isinstance(promo_message[1], InlineKeyboardMarkup)
        assert (
            promo_message[1].inline_keyboard == expected_markup.inline_keyboard
        )

    asyncio.run(scenario())


def test_start_command_blocked_during_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path)
        handler = get_message_handler(router, "handle_start")

        bot = DummyBot()
        message = DummyMessage(user_id=555, bot=bot)
        message.text = "/start"  # type: ignore[attr-defined]
        state = DummyState()
        await state.set_state(TryOnStates.GENERATING.state)
        await state.update_data(is_generating=True)

        await handler(message, state)

        assert message.answers[-1][0] == msg.WEAR_BUSY_MESSAGE
        assert state.state == TryOnStates.GENERATING.state

    asyncio.run(scenario())


def test_wear_sets_default_gender_when_missing(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, repository, _, _, _ = build_router(tmp_path)
        handler = get_message_handler(router, "command_wear")

        bot = DummyBot()
        message = DummyMessage(user_id=777, bot=bot)
        message.text = "/wear"  # type: ignore[attr-defined]
        state = DummyState()

        await handler(message, state)

        assert state.state is TryOnStates.AWAITING_PHOTO
        data = await state.get_data()
        assert data.get("gender") == "male"
        assert repository.updated_filters == [(777, "male")]
        assert message.answers[-1][0] == msg.PHOTO_INSTRUCTION

    asyncio.run(scenario())


def test_wear_rejected_during_contact_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path)
        handler = get_message_handler(router, "command_wear")

        bot = DummyBot()
        message = DummyMessage(user_id=778, bot=bot)
        message.text = "/wear"  # type: ignore[attr-defined]
        state = DummyState()
        await state.set_state(ContactRequest.waiting_for_phone.state)

        await handler(message, state)

        assert message.answers[-1][0] == msg.WEAR_BUSY_MESSAGE
        assert state.state == ContactRequest.waiting_for_phone.state

    asyncio.run(scenario())


def test_wear_rejected_during_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path)
        handler = get_message_handler(router, "command_wear")

        bot = DummyBot()
        message = DummyMessage(user_id=779, bot=bot)
        message.text = "/wear"  # type: ignore[attr-defined]
        state = DummyState()
        await state.set_state(TryOnStates.GENERATING.state)
        await state.update_data(is_generating=True)

        await handler(message, state)

        assert message.answers[-1][0] == msg.WEAR_BUSY_MESSAGE
        assert state.state == TryOnStates.GENERATING.state

    asyncio.run(scenario())


def test_help_allowed_during_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path)
        handler = get_message_handler(router, "command_help")

        bot = DummyBot()
        message = DummyMessage(user_id=888, bot=bot)
        message.text = "/help"  # type: ignore[attr-defined]
        state = DummyState()
        await state.set_state(TryOnStates.GENERATING.state)
        await state.update_data(is_generating=True)

        await handler(message, state)

        assert message.answers[-1][0] == msg.HELP_TEXT

    asyncio.run(scenario())


def test_privacy_command_ignores_non_photo_warning(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path)
        privacy_handler = get_message_handler(router, "command_privacy")
        fallback_handler = get_message_handler(router, "reject_non_photo")

        bot = DummyBot()
        message = DummyMessage(user_id=9990, bot=bot)
        message.text = "/privacy"  # type: ignore[attr-defined]
        state = DummyState()
        await state.set_state(TryOnStates.AWAITING_PHOTO.state)
        await state.update_data(is_generating=True)

        await privacy_handler(message, state)

        text, markup = message.answers[-1]
        assert text == msg.PRIVACY_POLICY_TEXT
        assert isinstance(markup, InlineKeyboardMarkup)
        button = markup.inline_keyboard[0][0]
        assert button.text == msg.MAIN_MENU_POLICY_BUTTON
        assert button.url == TEST_PRIVACY_POLICY_URL

        second_message = DummyMessage(user_id=9991, bot=bot)
        second_message.text = "/privacy"  # type: ignore[attr-defined]
        second_state = DummyState()
        await second_state.set_state(TryOnStates.AWAITING_PHOTO.state)

        await fallback_handler(second_message, second_state)

        assert second_message.answers == []

    asyncio.run(scenario())


def test_cancel_blocked_during_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, _, _, _, _ = build_router(tmp_path)
        handler = get_message_handler(router, "command_cancel")

        bot = DummyBot()
        message = DummyMessage(user_id=4321, bot=bot)
        message.text = "/cancel"  # type: ignore[attr-defined]
        state = DummyState()
        await state.set_state(TryOnStates.GENERATING.state)
        await state.update_data(is_generating=True)

        await handler(message, state)

        assert message.answers[-1][0] == msg.WEAR_BUSY_MESSAGE
        assert state.state == TryOnStates.GENERATING.state

    asyncio.run(scenario())


def test_cancel_resets_session(tmp_path: Path) -> None:
    async def scenario() -> None:
        router, repository, _, _, _ = build_router(tmp_path)
        handler = get_message_handler(router, "command_cancel")

        bot = DummyBot()
        message = DummyMessage(user_id=5432, bot=bot)
        message.text = "/cancel"  # type: ignore[attr-defined]
        state = DummyState()
        await state.set_state(TryOnStates.RESULT.state)
        await state.update_data(
            upload="/tmp/upload.jpg",
            upload_file_id="file123",
            selected_model="model",
            current_models=["model"],
            is_generating=False,
        )
        repository.more_buttons[5432] = (321, "result", {"site_url": "https://example.com"})

        await handler(message, state)

        assert message.answers[-1][0] == msg.CANCEL_CONFIRMATION
        assert state.state is TryOnStates.START
        data = await state.get_data()
        assert data.get("upload") is None
        assert data.get("upload_file_id") is None
        assert data.get("selected_model") is None
        assert data.get("current_models") == []
        assert data.get("is_generating") is False
        assert repository.more_buttons.get(5432) == (None, None, None)

    asyncio.run(scenario())


def test_no_attach_photo_text_in_sources() -> None:
    project_root = Path("app")
    matches = [
        path
        for path in project_root.rglob("*.py")
        if "Прикрепить фотку" in path.read_text(encoding="utf-8")
    ]
    assert matches == []
