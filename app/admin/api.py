"""Admin Mini App API server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aiohttp import web

from app.admin.data import list_admin_users
from app.admin.init_data import verify_init_data
from app.admin.security import is_admin
from app.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@web.middleware
async def cors_middleware(request: web.Request, handler) -> web.StreamResponse:
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
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
                "full_name": row.full_name,
                "telegram_link": row.telegram_link,
                "generations": row.generations,
                "site_clicks": row.site_clicks,
                "social_clicks": row.social_clicks,
                "phone": row.phone,
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


def create_app() -> web.Application:
    config = load_config()
    app = web.Application(middlewares=[cors_middleware])
    app["config"] = config
    app["db_path"] = (PROJECT_ROOT / "loop.db").resolve()
    app.router.add_route("GET", "/admin/api/users", handle_users)
    app.router.add_route("OPTIONS", "/admin/api/users", handle_users)
    return app


def main() -> None:
    host = os.getenv("ADMIN_API_HOST", "0.0.0.0")
    port = int(os.getenv("ADMIN_API_PORT", "8080"))
    web.run_app(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
