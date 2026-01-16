# Admin Mini App

This repo includes a minimal admin dashboard that is opened from the bot using `/admin`. The admin panel is now **fully integrated** into the main bot process — no separate deployment or configuration needed.

## Quick Start

1. **Add your Telegram ID to the admin whitelist:**
   Edit `app/admin/security.py` and add your Telegram user ID to `ADMIN_WHITELIST`.

2. **Set the admin panel URL in `.env`:**
   ```
   ADMIN_WEBAPP_URL=http://YOUR_VPS_IP:8080/admin
   ```
   Replace `YOUR_VPS_IP` with your server's public IP address or domain.

3. **Run the bot:**
   ```bash
   python manage.py run
   ```

4. **Access the admin panel:**
   Send `/admin` to your bot in Telegram.

That's it! The Admin API server starts automatically with the bot on port 8080.

## How it works

- The Admin API is embedded into the main bot process and starts automatically
- The API serves both the frontend (HTML/CSS/JS) and the data endpoints
- No need for Cloudflare Pages, tunnels, or separate processes
- Everything runs on a single port (default: 8080)

## Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ADMIN_WEBAPP_URL` | URL for the Telegram WebApp button | Required |
| `ADMIN_API_HOST` | Host to bind the API server | `0.0.0.0` |
| `ADMIN_API_PORT` | Port for the API server | `8080` |

### Admin whitelist

Edit `app/admin/security.py` to manage who can access the admin panel:

```python
ADMIN_WHITELIST: set[int] = {123456789, 987654321}  # Your Telegram user IDs
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /admin` | Admin panel UI |
| `GET /admin/api/users` | User data API (requires `X-Telegram-Init-Data`) |

## Security

- `/admin` command is not shown in the bot menu
- Non-admin users get no response when using `/admin`
- API requires valid Telegram `initData` and checks the admin whitelist
- CORS is enabled for Telegram WebApp compatibility

## Advanced: Standalone API server

If you need to run the API separately (not recommended):

```bash
python manage.py admin-api
```

## Troubleshooting

**"Server unavailable" in the admin panel:**
- Check that `ADMIN_WEBAPP_URL` points to your server's public IP/domain
- Ensure port 8080 is open in your firewall
- Verify the bot is running

**No response to `/admin` command:**
- Add your Telegram user ID to `ADMIN_WHITELIST` in `app/admin/security.py`

**Empty user list:**
- The database may be empty if no users have interacted with the bot yet
