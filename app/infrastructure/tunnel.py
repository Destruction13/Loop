"""Cloudflare Tunnel manager for automatic HTTPS exposure of the Admin API.

This module provides functionality to:
1. Start cloudflared tunnel automatically with the bot
2. Parse the generated tunnel URL
3. Update the frontend config.js with the correct API URL

Usage:
    tunnel = CloudflareTunnel(local_port=8080)
    await tunnel.start()
    print(tunnel.public_url)  # https://xxx-xxx-xxx.trycloudflare.com
    ...
    await tunnel.stop()
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Callable

from logger import get_logger

logger = get_logger(__name__)

# Path to the miniapps/admin directory for config.js updates
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MINIAPP_CONFIG_PATH = PROJECT_ROOT / "miniapps" / "admin" / "config.js"


class CloudflareTunnel:
    """Manages a cloudflared quick tunnel for exposing local services via HTTPS."""

    def __init__(
        self,
        local_port: int = 8080,
        local_host: str = "localhost",
        on_url_ready: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the tunnel manager.

        Args:
            local_port: Local port to expose (default: 8080)
            local_host: Local host to expose (default: localhost)
            on_url_ready: Optional callback when tunnel URL is ready
        """
        self.local_port = local_port
        self.local_host = local_host
        self.on_url_ready = on_url_ready
        self._process: asyncio.subprocess.Process | None = None
        self._public_url: str | None = None
        self._started = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None

    @property
    def public_url(self) -> str | None:
        """Get the public HTTPS URL of the tunnel."""
        return self._public_url

    @property
    def is_running(self) -> bool:
        """Check if the tunnel is currently running."""
        return self._process is not None and self._process.returncode is None

    @staticmethod
    def is_available() -> bool:
        """Check if cloudflared is installed and available."""
        return shutil.which("cloudflared") is not None

    async def start(self, timeout: float = 30.0) -> str | None:
        """Start the cloudflared tunnel and wait for the URL.

        Args:
            timeout: Maximum time to wait for tunnel URL (seconds)

        Returns:
            The public HTTPS URL, or None if failed
        """
        if not self.is_available():
            logger.error(
                "cloudflared is not installed. Install it from: "
                "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            )
            return None

        if self.is_running:
            logger.warning("Tunnel is already running")
            return self._public_url

        try:
            # Start cloudflared with quick tunnel (no account needed)
            self._process = await asyncio.create_subprocess_exec(
                "cloudflared",
                "tunnel",
                "--url",
                f"http://{self.local_host}:{self.local_port}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Start monitoring task to read output and find URL
            self._monitor_task = asyncio.create_task(self._monitor_output())

            # Wait for URL with timeout
            try:
                await asyncio.wait_for(self._started.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for tunnel URL")
                await self.stop()
                return None

            if self._public_url:
                logger.info(f"Cloudflare Tunnel started: {self._public_url}")
                if self.on_url_ready:
                    self.on_url_ready(self._public_url)
                return self._public_url

        except Exception as exc:
            logger.error(f"Failed to start cloudflared tunnel: {exc}")
            await self.stop()

        return None

    async def _monitor_output(self) -> None:
        """Monitor cloudflared output to extract the tunnel URL."""
        if not self._process or not self._process.stderr:
            return

        # Pattern to match the tunnel URL in cloudflared output
        url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break

                decoded = line.decode("utf-8", errors="ignore").strip()
                if decoded:
                    logger.debug(f"cloudflared: {decoded}")

                # Look for the tunnel URL
                match = url_pattern.search(decoded)
                if match and not self._public_url:
                    self._public_url = match.group(0)
                    self._started.set()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error monitoring cloudflared output: {exc}")

    async def stop(self) -> None:
        """Stop the cloudflared tunnel."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as exc:
                logger.error(f"Error stopping cloudflared: {exc}")
            finally:
                self._process = None

        self._public_url = None
        self._started.clear()
        logger.info("Cloudflare Tunnel stopped")

    async def wait_closed(self) -> None:
        """Wait for the tunnel process to exit."""
        if self._process:
            await self._process.wait()


def update_admin_config_js(api_base_url: str) -> bool:
    """Update the miniapps/admin/config.js with the new API URL.

    Args:
        api_base_url: The public HTTPS URL of the API

    Returns:
        True if successful, False otherwise
    """
    try:
        config_content = f'''window.ADMIN_CONFIG = {{
  apiBaseUrl: "{api_base_url}",
}};
'''
        MINIAPP_CONFIG_PATH.write_text(config_content, encoding="utf-8")
        logger.info(f"Updated config.js with API URL: {api_base_url}")
        return True
    except Exception as exc:
        logger.error(f"Failed to update config.js: {exc}")
        return False


async def start_tunnel_and_update_config(
    local_port: int = 8080,
    timeout: float = 30.0,
) -> tuple[CloudflareTunnel | None, str | None]:
    """Convenience function to start tunnel.

    Note: config.js update is no longer needed because the frontend is now
    served from the tunnel itself, so apiBaseUrl can remain empty (same origin).

    Args:
        local_port: Local port where Admin API is running
        timeout: Timeout for tunnel startup

    Returns:
        Tuple of (tunnel instance, public URL) or (None, None) if failed
    """
    if not CloudflareTunnel.is_available():
        logger.warning(
            "cloudflared is not installed. Admin API will not be accessible via HTTPS. "
            "Install cloudflared for automatic tunnel: "
            "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        )
        return None, None

    tunnel = CloudflareTunnel(local_port=local_port)
    url = await tunnel.start(timeout=timeout)

    if url:
        return tunnel, url

    return None, None
