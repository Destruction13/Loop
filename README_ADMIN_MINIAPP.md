# Admin Mini App (Telegram + Cloudflare Pages)

This repo includes a minimal admin dashboard that is opened from the bot using `/admin`. The UI is a static Mini App hosted on Cloudflare Pages and pulls data from a lightweight API.

## 1) Configure access

- Update `app/admin/security.py` and set your real Telegram user IDs in `ADMIN_WHITELIST`.
- `/admin` is intentionally **not** added to the bot command menu.

## 2) Environment variables

Add these to `.env` (they are already in `.env.example`):

```
ADMIN_WEBAPP_URL=https://your-project.pages.dev
ADMIN_API_BASE_URL=https://your-api-host
```

- `ADMIN_WEBAPP_URL` is used by the bot to open the Mini App.
- `ADMIN_API_BASE_URL` is used by the frontend config (see below).

## 3) Run the admin API

The API is a small aiohttp server:

```
python manage.py admin-api
```

Optional overrides:

```
ADMIN_API_HOST=0.0.0.0
ADMIN_API_PORT=8080
```

The endpoint is:

```
GET /admin/api/users?offset=&limit=&sort=&order=
```

It requires `X-Telegram-Init-Data` and checks the admin whitelist.

## 4) Configure the Mini App frontend

Edit `miniapps/admin/config.js` and set:

```
window.ADMIN_CONFIG = {
  apiBaseUrl: "https://your-api-host",
};
```

If the API is served from the same origin as the Mini App, `apiBaseUrl` can be empty.

## 5) Cloudflare Pages deploy (static)

1. Create a new Pages project from this repo.
2. **Root directory**: `miniapps/admin`
3. **Build command**: leave empty
4. **Build output directory**: `.`
5. Deploy, then copy the Pages URL into `ADMIN_WEBAPP_URL`.

No `wrangler.toml` is required for this static build.

## 6) Manual checks

- Non-admin `/admin` -> no hints or response.
- Admin `/admin` -> button opens the Mini App.
- Mini App inside Telegram loads the table.
- Opening the Mini App URL in a browser works, but API returns `403` without initData.
