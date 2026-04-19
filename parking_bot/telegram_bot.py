from __future__ import annotations

from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import Settings
from parking_bot.types import Availability, CameraStatusRecord, Subscription, TimeWindow


WINDOW_STATE_KEY = "awaiting_window_camera_id"


def build_application(settings: Settings, service: CameraMonitorService) -> Application:
    async def post_init(application: Application) -> None:
        application.bot_data["service"] = service
        await service.start(application.bot)

    async def post_shutdown(application: Application) -> None:
        monitor = application.bot_data.get("service")
        if monitor is not None:
            await monitor.stop()

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(8)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("cameras", cameras_command))
    app.add_handler(CommandHandler("subscriptions", subscriptions_command))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    return app


def _get_service(application_or_context: Application | ContextTypes.DEFAULT_TYPE) -> CameraMonitorService:
    if isinstance(application_or_context, Application):
        return application_or_context.bot_data["service"]
    return application_or_context.application.bot_data["service"]


def _get_repository(context: ContextTypes.DEFAULT_TYPE) -> SQLiteRepository:
    return _get_service(context).repository


async def _reply_camera_unavailable(
    context: ContextTypes.DEFAULT_TYPE,
    message,
) -> None:
    if message is None:
        return
    await message.reply_text(
        "Эта камера больше недоступна. Откройте актуальный список камер и выберите нужную заново.",
        reply_markup=_main_keyboard(_get_service(context)),
    )


def _chunks(items: list[InlineKeyboardButton], size: int) -> list[list[InlineKeyboardButton]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _main_keyboard(service: CameraMonitorService) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(camera.display_name, callback_data=f"camera:{camera.id}")
        for camera in service.cameras
    ]
    rows = _chunks(buttons, 2)
    rows.append([InlineKeyboardButton("Мои подписки", callback_data="subs:list")])
    return InlineKeyboardMarkup(rows)


def _back_to_cameras_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Назад к камерам", callback_data="nav:cameras")]]
    )


def _camera_actions(camera_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Подписка 24/7", callback_data=f"sub:{camera_id}"),
                InlineKeyboardButton("Подписка по времени", callback_data=f"window:{camera_id}"),
            ],
            [
                InlineKeyboardButton("Назад", callback_data="nav:cameras"),
                InlineKeyboardButton("Обновить", callback_data=f"refresh:{camera_id}"),
            ],
            [
                InlineKeyboardButton("Мои подписки", callback_data="subs:list"),
            ],
        ]
    )


def _subscriptions_keyboard(subscriptions: list[Subscription]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(f"Отписаться #{subscription.id}", callback_data=f"unsub:{subscription.id}")]
        for subscription in subscriptions
    ]
    rows.append([InlineKeyboardButton("Назад к камерам", callback_data="nav:cameras")])
    return InlineKeyboardMarkup(rows)


def _status_label(status: Availability) -> str:
    mapping = {
        Availability.FREE: "обнаружены свободные места",
        Availability.FULL: "свободных мест не видно",
        Availability.UNKNOWN: "состояние не определено",
    }
    return mapping[status]


def _status_text(service: CameraMonitorService, status: CameraStatusRecord | None) -> str:
    if status is None or status.observed_at is None:
        return "Для этой камеры ещё нет обработанного кадра."

    observed = status.observed_at.astimezone(service.timezone).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Камера: {status.display_name}",
        f"Стабильное состояние: {_status_label(status.stable_availability)}",
        f"Текущий кадр: {_status_label(status.current_availability)}",
        f"Свободных мест: {status.free_count}",
        f"Занятых мест: {status.occupied_count}",
        f"Обработано: {observed}",
    ]
    if status.source_updated_at is not None:
        source_updated = status.source_updated_at.astimezone(service.timezone).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        lines.append(f"Кадр источника: {source_updated}")
        lag_seconds = int((status.observed_at - status.source_updated_at).total_seconds())
        if lag_seconds > 120:
            lag_minutes = lag_seconds // 60
            lines.append(
                f"Источник Ufanet отдаёт кэшированный кадр, задержка около {lag_minutes} мин."
            )
    if status.last_error:
        lines.append(f"Последняя ошибка: {status.last_error}")
    return "\n".join(lines)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = _get_service(context)
    text = (
        "Бот уведомлений о свободных парковочных местах\n\n"
        "Выберите камеру, посмотрите её текущее состояние и оформите подписку "
        "на уведомления о появлении свободного места."
    )
    await update.effective_message.reply_text(text, reply_markup=_main_keyboard(service))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Команды:\n"
        "/start - открыть список камер\n"
        "/cameras - показать камеры\n"
        "/subscriptions - показать активные подписки\n\n"
        "Чтобы подписаться на определённое время, нажмите 'Подписка по времени' "
        "и отправьте интервал в формате 18:00-19:00."
    )
    await update.effective_message.reply_text(text)


async def cameras_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_cameras_menu(update, context)


async def subscriptions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    repository = _get_repository(context)
    service = _get_service(context)
    subscriptions = repository.list_subscriptions(update.effective_chat.id)
    if not subscriptions:
        await update.effective_message.reply_text(
            "У вас пока нет активных подписок.",
            reply_markup=_back_to_cameras_keyboard(),
        )
        return

    lines = ["Активные подписки:"]
    for subscription in subscriptions:
        camera = service.find_camera(subscription.camera_id)
        window_text = (
            subscription.time_window.format_for_humans() if subscription.time_window else "24/7"
        )
        camera_name = (
            camera.display_name
            if camera is not None
            else f"{subscription.camera_id} (камера больше недоступна)"
        )
        lines.append(f"#{subscription.id}: {camera_name} | {window_text}")

    await update.effective_message.reply_text(
        "\n".join(lines),
        reply_markup=_subscriptions_keyboard(subscriptions),
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    service = _get_service(context)
    try:
        await query.answer()
    except BadRequest:
        pass

    data = query.data or ""
    if data.startswith("camera:"):
        camera_id = data.split(":", 1)[1]
        if service.find_camera(camera_id) is None:
            await _reply_camera_unavailable(context, query.message)
            return
        await _show_cached_camera_status(update, context, camera_id)
        return

    if data == "nav:cameras":
        await _send_cameras_menu(update, context)
        return

    if data.startswith("refresh:"):
        camera_id = data.split(":", 1)[1]
        if service.find_camera(camera_id) is None:
            await _reply_camera_unavailable(context, query.message)
            return
        await _queue_camera_refresh(update, context, camera_id)
        return

    if data.startswith("sub:"):
        camera_id = data.split(":", 1)[1]
        if service.find_camera(camera_id) is None:
            await _reply_camera_unavailable(context, query.message)
            return
        await _create_subscription(update, context, camera_id, time_window=None)
        return

    if data.startswith("window:"):
        camera_id = data.split(":", 1)[1]
        camera = service.find_camera(camera_id)
        if camera is None:
            await _reply_camera_unavailable(context, query.message)
            return
        context.user_data[WINDOW_STATE_KEY] = camera_id
        await query.message.reply_text(
            f"Отправьте ежедневный интервал для камеры {camera.display_name} в формате ЧЧ:ММ-ЧЧ:ММ.\n"
            "Пример: 18:00-19:00\n"
            "Чтобы вернуться, напишите 'назад'."
        )
        return

    if data == "subs:list":
        await subscriptions_command(update, context)
        return

    if data.startswith("unsub:"):
        subscription_id = int(data.split(":", 1)[1])
        repository = _get_repository(context)
        deleted = repository.deactivate_subscription(update.effective_chat.id, subscription_id)
        message = "Подписка удалена." if deleted else "Подписка не найдена."
        await query.message.reply_text(message, reply_markup=_back_to_cameras_keyboard())


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pending_camera_id = context.user_data.get(WINDOW_STATE_KEY)
    if not pending_camera_id:
        return

    service = _get_service(context)
    if service.find_camera(pending_camera_id) is None:
        context.user_data.pop(WINDOW_STATE_KEY, None)
        await _reply_camera_unavailable(context, update.effective_message)
        return

    text = update.effective_message.text.strip()
    if text.lower() in {"cancel", "stop", "back", "назад"}:
        context.user_data.pop(WINDOW_STATE_KEY, None)
        await update.effective_message.reply_text(
            "Настройка подписки по времени отменена.",
            reply_markup=_main_keyboard(_get_service(context)),
        )
        return

    try:
        window = TimeWindow.parse(text)
    except ValueError:
        await update.effective_message.reply_text(
            "Не удалось распознать интервал. Пример: 18:00-19:00"
        )
        return

    context.user_data.pop(WINDOW_STATE_KEY, None)
    await _create_subscription(update, context, pending_camera_id, time_window=window)


async def _show_cached_camera_status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    camera_id: str,
) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return

    service = _get_service(context)
    camera = service.find_camera(camera_id)
    if camera is None:
        await _reply_camera_unavailable(context, query.message)
        return

    status = service.get_status(camera_id)
    if status is None:
        await query.message.reply_text(
            f"Подготавливаю первый кадр для камеры {camera.display_name}. "
            "Это может занять до минуты, пришлю результат следующим сообщением."
        )
        context.application.create_task(
            _send_fresh_camera_status(context, update.effective_chat.id, camera_id)
        )
        return

    await _send_camera_status_message(context, update.effective_chat.id, camera_id, status)


async def _send_cameras_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    service = _get_service(context)
    message = update.effective_message or (update.callback_query.message if update.callback_query else None)
    if message is None:
        return
    await message.reply_text(
        "Доступные камеры:",
        reply_markup=_main_keyboard(service),
    )


async def _queue_camera_refresh(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    camera_id: str,
) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return

    camera = _get_service(context).find_camera(camera_id)
    if camera is None:
        await _reply_camera_unavailable(context, query.message)
        return
    await query.message.reply_text(
        f"Обновляю камеру {camera.display_name}. "
        "Свежий кадр пришлю следующим сообщением."
    )
    context.application.create_task(
        _send_fresh_camera_status(context, update.effective_chat.id, camera_id)
    )


async def _send_fresh_camera_status(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    camera_id: str,
) -> None:
    service = _get_service(context)
    camera = service.find_camera(camera_id)
    if camera is None:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Эта камера больше недоступна. Откройте /cameras и выберите актуальную заново.",
            reply_markup=_main_keyboard(service),
        )
        return
    try:
        status = await service.refresh_camera(camera_id, notify=False)
    except Exception as exc:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Не удалось обновить камеру {camera.display_name}: {exc}",
        )
        return

    await _send_camera_status_message(context, chat_id, camera_id, status)


async def _send_camera_status_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    camera_id: str,
    status: CameraStatusRecord,
) -> None:
    service = _get_service(context)
    camera = service.find_camera(camera_id)
    if camera is None:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Эта камера больше недоступна. Откройте /cameras и выберите актуальную заново.",
            reply_markup=_main_keyboard(service),
        )
        return
    text = _status_text(service, status)
    preview = status.annotated_frame_path or status.raw_frame_path

    if preview and preview.exists():
        with preview.open("rb") as frame:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=frame,
                caption=text,
                reply_markup=_camera_actions(camera.id),
            )
        return

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=_camera_actions(camera.id),
    )


async def _create_subscription(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    camera_id: str,
    *,
    time_window: TimeWindow | None,
) -> None:
    message = update.effective_message or (update.callback_query.message if update.callback_query else None)
    if message is None:
        return

    repository = _get_repository(context)
    service = _get_service(context)
    camera = service.find_camera(camera_id)
    if camera is None:
        await _reply_camera_unavailable(context, message)
        return
    subscription = repository.upsert_subscription(
        chat_id=update.effective_chat.id,
        camera_id=camera_id,
        time_window=time_window,
        created_at=datetime.now(tz=service.timezone),
    )
    window_text = time_window.format_for_humans() if time_window else "24/7"
    await message.reply_text(
        f"Подписка сохранена.\n"
        f"Камера: {camera.display_name}\n"
        f"Интервал: {window_text}\n"
        f"Номер подписки: {subscription.id}",
        reply_markup=_camera_actions(camera_id),
    )
