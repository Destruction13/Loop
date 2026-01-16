"""Admin Mini App API server."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

from aiohttp import web

from app.admin.data import list_admin_users
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


def create_app(config: Config | None = None, db_path: Path | None = None) -> web.Application:
    """Create the admin API aiohttp application.
    
    Args:
        config: Application config. If None, will be loaded from environment.
        db_path: Path to SQLite database. If None, uses default location.
    """
    if config is None:
        config = load_config()
    app = web.Application(middlewares=[cors_middleware])
    app["config"] = config
    app["db_path"] = db_path or (PROJECT_ROOT / "loop.db").resolve()
    
    # API endpoints
    app.router.add_route("GET", "/admin/api/users", handle_users)
    app.router.add_route("OPTIONS", "/admin/api/users", handle_users)
    
    # Static files for admin panel (served from same origin)
    app.router.add_get("/admin", handle_admin_root)
    app.router.add_get("/admin/", handle_admin_root)
    app.router.add_get("/admin/config.js", handle_config_js)
    app.router.add_get("/admin/{filename}", handle_static_file)
    
    return app


async def start_admin_api(
    config: Config,
    db_path: Path,
    *,
    host: str | None = None,
    port: int | None = None,
) -> tuple[web.AppRunner, web.TCPSite]:
    """Start the admin API server as part of the main bot process.
    
    Returns:
        Tuple of (runner, site) for cleanup on shutdown.
    """
    resolved_host = host or os.getenv("ADMIN_API_HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("ADMIN_API_PORT", "8080"))
    
    app = create_app(config=config, db_path=db_path)
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
