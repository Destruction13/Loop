"""Admin Mini App API server."""

from __future__ import annotations

import mimetypes
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from aiohttp import web

from datetime import datetime, timedelta, timezone

from app.admin.data import (
    delete_user,
    get_dashboard_data,
    get_stats,
    get_user_details,
    list_admin_users,
    update_event_tries,
    update_user_tries,
)
from app.admin.init_data import verify_init_data
from app.admin.security import is_admin
from app.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MINIAPP_DIR = PROJECT_ROOT / "miniapps" / "admin"


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Telegram-Init-Data"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


def _extract_init_data(request: web.Request) -> str | None:
    header = request.headers.get("X-Telegram-Init-Data")
    if header:
        return header.strip()
    return None


async def handle_users(request: web.Request) -> web.Response:
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)

    params = request.rel_url.query

    def _to_int(value: str | None, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    offset = _to_int(params.get("offset"), 0)
    limit = _to_int(params.get("limit"), 50)
    sort = params.get("sort", "generations") or "generations"
    order = params.get("order", "desc") or "desc"

    items, total = list_admin_users(
        request.app["db_path"],
        offset=offset,
        limit=limit,
        sort=sort,
        order=order,
    )
    payload: list[dict[str, Any]] = []
    for row in items:
        payload.append(
            {
                "user_id": row.user_id,
                "username": row.username,
                "first_name": row.first_name,
                "last_name": row.last_name,
                "full_name": row.full_name,
                "display_name": row.display_name,
                "telegram_link": row.telegram_link,
                "generations": row.generations,
                "tries_used": row.tries_used,
                "tries_limit": row.tries_limit,
                "tries_remaining": row.tries_remaining,
                "site_clicks": row.site_clicks,
                "social_clicks": row.social_clicks,
                "phone": row.phone,
                "event_paid_used": row.event_paid_used,
                "event_paid_limit": row.event_paid_limit,
                "event_paid_remaining": row.event_paid_remaining,
                "has_event_record": row.has_event_record,
                "event_id": row.event_id,
            }
        )

    return web.json_response(
        {
            "items": payload,
            "total": total,
            "offset": offset,
            "limit": limit,
        }
    )


async def handle_stats(request: web.Request) -> web.Response:
    """Get overall statistics."""
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)

    stats = get_stats(request.app["db_path"])
    return web.json_response(stats)


async def handle_user_details(request: web.Request) -> web.Response:
    """Get detailed info about a single user."""
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        user_id = int(request.match_info["user_id"])
    except (KeyError, ValueError):
        return web.json_response({"error": "invalid_user_id"}, status=400)

    details = get_user_details(request.app["db_path"], user_id)
    if details is None:
        return web.json_response({"error": "user_not_found"}, status=404)

    return web.json_response(details)


async def handle_delete_user(request: web.Request) -> web.Response:
    """Delete a user completely."""
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        user_id = int(request.match_info["user_id"])
    except (KeyError, ValueError):
        return web.json_response({"error": "invalid_user_id"}, status=400)

    success = delete_user(request.app["db_path"], user_id)
    if not success:
        return web.json_response({"error": "user_not_found"}, status=404)

    return web.json_response({"ok": True, "deleted_user_id": user_id})


async def handle_update_tries(request: web.Request) -> web.Response:
    """Update user's tries (remaining or limit)."""
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        user_id = int(request.match_info["user_id"])
    except (KeyError, ValueError):
        return web.json_response({"error": "invalid_user_id"}, status=400)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    tries_remaining = body.get("tries_remaining")
    tries_limit = body.get("tries_limit")

    if tries_remaining is not None:
        try:
            tries_remaining = int(tries_remaining)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_tries_remaining"}, status=400)

    if tries_limit is not None:
        try:
            tries_limit = int(tries_limit)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_tries_limit"}, status=400)

    success = update_user_tries(
        request.app["db_path"],
        user_id,
        tries_remaining=tries_remaining,
        tries_limit=tries_limit,
    )
    if not success:
        return web.json_response({"error": "update_failed"}, status=400)

    return web.json_response({"ok": True})


async def handle_update_event_tries(request: web.Request) -> web.Response:
    """Update user's event attempts."""
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        user_id = int(request.match_info["user_id"])
        event_id = request.match_info["event_id"]
    except (KeyError, ValueError):
        return web.json_response({"error": "invalid_parameters"}, status=400)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    paid_used = body.get("paid_used")

    if paid_used is not None:
        try:
            paid_used = int(paid_used)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_paid_used"}, status=400)

    success = update_event_tries(
        request.app["db_path"],
        user_id,
        event_id,
        paid_used=paid_used,
    )
    if not success:
        return web.json_response({"error": "update_failed"}, status=400)

    return web.json_response({"ok": True})


async def handle_config_js(request: web.Request) -> web.Response:
    """Serve dynamically generated config.js with the correct API base URL."""
    # The API base URL is the same origin as the request
    # Since we serve both frontend and API from the same server, use empty string
    config_content = """window.ADMIN_CONFIG = {
  apiBaseUrl: "",
};
"""
    return web.Response(
        text=config_content,
        content_type="application/javascript",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )


async def handle_static_file(request: web.Request) -> web.Response:
    """Serve static files from the miniapps/admin directory."""
    filename = request.match_info.get("filename", "index.html")
    if not filename or filename == "":
        filename = "index.html"
    
    # Security: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        return web.Response(status=404, text="Not Found")
    
    # Skip config.js - it's handled by handle_config_js
    if filename == "config.js":
        return await handle_config_js(request)
    
    file_path = MINIAPP_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        return web.Response(status=404, text="Not Found")
    
    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type is None:
        content_type = "application/octet-stream"
    
    return web.FileResponse(file_path, headers={"Content-Type": content_type})


async def handle_admin_root(request: web.Request) -> web.Response:
    """Serve the admin panel index.html."""
    index_path = MINIAPP_DIR / "index.html"
    if not index_path.exists():
        return web.Response(status=404, text="Admin panel not found")
    return web.FileResponse(index_path, headers={"Content-Type": "text/html"})


async def handle_dashboard(request: web.Request) -> web.Response:
    """Get dashboard data for active users chart.
    
    Query params:
        period: Time period - "4h", "8h", "24h", "3d", "7d" (default: "24h")
    
    Returns:
        JSON with chart data points containing:
        - time: ISO timestamp for the time slot
        - active_users: Number of unique users with activity in that slot
        - total_actions: Total number of actions in that slot (optional metric)
    """
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)
    
    # Parse period parameter
    period = request.rel_url.query.get("period", "24h")
    valid_periods = {"4h", "8h", "24h", "3d", "7d"}
    if period not in valid_periods:
        return web.json_response({"error": "invalid_period"}, status=400)
    
    # Calculate time range
    now = datetime.now(timezone.utc)
    period_hours = {
        "4h": 4,
        "8h": 8,
        "24h": 24,
        "3d": 72,
        "7d": 168,
    }
    hours = period_hours[period]
    start_time = now - timedelta(hours=hours)
    
    # Determine aggregation: hourly for <=24h, daily for >24h
    use_daily = hours > 24
    
    db_path: Path = request.app["db_path"]
    data = get_dashboard_data(db_path, start_time, now, use_daily=use_daily)
    
    return web.json_response({
        "period": period,
        "aggregation": "daily" if use_daily else "hourly",
        "data": data,
    })


async def handle_broadcast(request: web.Request) -> web.Response:
    """Send a broadcast message to all users.
    
    Supports:
    - Text message with HTML formatting
    - Photo with caption (photo as base64 or URL)
    - Emoji and special characters
    """
    from aiogram import Bot
    from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest
    from aiogram.types import BufferedInputFile
    import base64
    import asyncio
    
    init_data = _extract_init_data(request)
    config = request.app["config"]
    if not init_data:
        return web.json_response({"error": "init_data_required"}, status=403)
    verified = verify_init_data(init_data, config.bot_token)
    if verified is None or not is_admin(verified.user_id):
        return web.json_response({"error": "forbidden"}, status=403)
    
    bot: Bot | None = request.app.get("bot")
    if bot is None:
        return web.json_response({"error": "bot_not_available"}, status=500)
    
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    
    text = body.get("text", "").strip()
    photo_base64 = body.get("photo_base64", "").strip()
    photo_url = body.get("photo_url", "").strip()
    
    if not text and not photo_base64 and not photo_url:
        return web.json_response({"error": "message_required"}, status=400)
    
    # Get all user IDs from database
    db_path: Path = request.app["db_path"]
    user_ids: list[int] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT user_id FROM users")
            user_ids = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        return web.json_response({"error": f"db_error: {e}"}, status=500)
    
    if not user_ids:
        return web.json_response({"error": "no_users"}, status=400)
    
    # Prepare photo if provided
    photo_input = None
    if photo_base64:
        try:
            photo_bytes = base64.b64decode(photo_base64)
            photo_input = BufferedInputFile(photo_bytes, filename="broadcast.jpg")
        except Exception:
            return web.json_response({"error": "invalid_photo_base64"}, status=400)
    elif photo_url:
        photo_input = photo_url
    
    # Send messages
    sent = 0
    failed = 0
    blocked = 0
    
    # Helper function to send message with HTML fallback to plain text
    async def send_to_user(uid: int, is_first: bool) -> str:
        """Send message to user. Returns: 'sent', 'blocked', 'failed'."""
        nonlocal photo_input
        
        # Try HTML first, then plain text on parse error
        for parse_mode in ("HTML", None):
            try:
                if photo_input:
                    # For photo: need to recreate BufferedInputFile after first use
                    if isinstance(photo_input, BufferedInputFile) and not is_first:
                        photo_input = BufferedInputFile(
                            base64.b64decode(photo_base64), 
                            filename="broadcast.jpg"
                        )
                    await bot.send_photo(
                        chat_id=uid,
                        photo=photo_input,
                        caption=text or None,
                        parse_mode=parse_mode,
                    )
                else:
                    await bot.send_message(
                        chat_id=uid,
                        text=text,
                        parse_mode=parse_mode,
                    )
                return "sent"
            except TelegramBadRequest as e:
                # If HTML parsing failed, retry without parse_mode
                if parse_mode == "HTML" and "can't parse" in str(e).lower():
                    continue
                return "failed"
            except TelegramForbiddenError:
                return "blocked"
            except Exception:
                return "failed"
        return "failed"
    
    is_first = True
    for user_id in user_ids:
        result = await send_to_user(user_id, is_first)
        is_first = False
        if result == "sent":
            sent += 1
        elif result == "blocked":
            blocked += 1
        else:
            failed += 1
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.05)
    
    return web.json_response({
        "ok": True,
        "total": len(user_ids),
        "sent": sent,
        "blocked": blocked,
        "failed": failed,
    })


async def handle_redirect(request: web.Request) -> web.Response:
    """Handle redirect with click tracking.
    
    URL format: /r/{user_id}?url={encoded_target_url}
    
    This endpoint:
    1. Increments the site_clicks counter for the user in the database
    2. Redirects the user to the target URL
    """
    user_id_str = request.match_info.get("user_id", "")
    target_url = request.rel_url.query.get("url", "")
    
    # Decode URL if it was encoded
    if target_url:
        target_url = unquote(target_url)
    
    # Validate user_id
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        user_id = None
    
    # Increment site_clicks if user_id is valid
    if user_id is not None:
        db_path: Path = request.app["db_path"]
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE users SET site_clicks = COALESCE(site_clicks, 0) + 1 WHERE user_id = ?",
                    (user_id,),
                )
                conn.commit()
        except Exception:
            pass  # Don't fail redirect if DB update fails
    
    # Redirect to target URL or fallback
    if not target_url:
        # Fallback to a default URL if none provided
        target_url = "https://loov.ru"
    
    raise web.HTTPFound(location=target_url)


def create_app(
    config: Config | None = None,
    db_path: Path | None = None,
    bot: Any = None,
) -> web.Application:
    """Create the admin API aiohttp application.
    
    Args:
        config: Application config. If None, will be loaded from environment.
        db_path: Path to SQLite database. If None, uses default location.
        bot: aiogram Bot instance for broadcast functionality.
    """
    if config is None:
        config = load_config()
    # Increase max body size to 16MB to support base64 photo uploads
    app = web.Application(middlewares=[cors_middleware], client_max_size=16 * 1024 * 1024)
    app["config"] = config
    app["db_path"] = db_path or (PROJECT_ROOT / "loop.db").resolve()
    app["bot"] = bot
    
    # API endpoints
    app.router.add_route("GET", "/admin/api/users", handle_users)
    app.router.add_route("OPTIONS", "/admin/api/users", handle_users)
    app.router.add_route("GET", "/admin/api/stats", handle_stats)
    app.router.add_route("OPTIONS", "/admin/api/stats", handle_stats)
    app.router.add_route("GET", "/admin/api/users/{user_id}", handle_user_details)
    app.router.add_route("OPTIONS", "/admin/api/users/{user_id}", handle_user_details)
    app.router.add_route("DELETE", "/admin/api/users/{user_id}", handle_delete_user)
    app.router.add_route("POST", "/admin/api/users/{user_id}/tries", handle_update_tries)
    app.router.add_route("OPTIONS", "/admin/api/users/{user_id}/tries", handle_update_tries)
    app.router.add_route("POST", "/admin/api/users/{user_id}/events/{event_id}", handle_update_event_tries)
    app.router.add_route("OPTIONS", "/admin/api/users/{user_id}/events/{event_id}", handle_update_event_tries)
    
    # Broadcast endpoint
    app.router.add_route("POST", "/admin/api/broadcast", handle_broadcast)
    app.router.add_route("OPTIONS", "/admin/api/broadcast", handle_broadcast)
    
    # Dashboard endpoint
    app.router.add_route("GET", "/admin/api/dashboard", handle_dashboard)
    app.router.add_route("OPTIONS", "/admin/api/dashboard", handle_dashboard)
    
    # Static files for admin panel (served from same origin)
    app.router.add_get("/admin", handle_admin_root)
    app.router.add_get("/admin/", handle_admin_root)
    app.router.add_get("/admin/config.js", handle_config_js)
    app.router.add_get("/admin/{filename}", handle_static_file)
    
    # Redirect endpoint for tracking clicks on "Подробнее о модели" button
    app.router.add_get("/r/{user_id}", handle_redirect)
    
    return app


async def start_admin_api(
    config: Config,
    db_path: Path,
    *,
    host: str | None = None,
    port: int | None = None,
    bot: Any = None,
) -> tuple[web.AppRunner, web.TCPSite]:
    """Start the admin API server as part of the main bot process.
    
    Args:
        config: Application config.
        db_path: Path to SQLite database.
        host: Host to bind to. Defaults to ADMIN_API_HOST env var or "0.0.0.0".
        port: Port to bind to. Defaults to ADMIN_API_PORT env var or 8080.
        bot: aiogram Bot instance for broadcast functionality.
    
    Returns:
        Tuple of (runner, site) for cleanup on shutdown.
    """
    resolved_host = host or os.getenv("ADMIN_API_HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("ADMIN_API_PORT", "8080"))
    
    app = create_app(config=config, db_path=db_path, bot=bot)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, resolved_host, resolved_port)
    await site.start()
    return runner, site


async def stop_admin_api(runner: web.AppRunner) -> None:
    """Stop the admin API server gracefully."""
    await runner.cleanup()


def main() -> None:
    host = os.getenv("ADMIN_API_HOST", "0.0.0.0")
    port = int(os.getenv("ADMIN_API_PORT", "8080"))
    web.run_app(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
