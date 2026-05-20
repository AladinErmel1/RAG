from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import msal
import requests


GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
DEFAULT_SCOPES = ["User.Read", "Files.Read", "offline_access"]


def download_from_onedrive(
    *,
    client_id: str,
    tenant_id: str,
    remote_path: str,
    destination_dir: Path,
    extensions: set[str],
    max_files: int,
    token_cache_path: Path,
    device_flow_callback: Callable[[dict], None] | None = None,
) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    token_cache_path.parent.mkdir(parents=True, exist_ok=True)
    token = _get_token(
        client_id=client_id,
        tenant_id=tenant_id,
        token_cache_path=token_cache_path,
        device_flow_callback=device_flow_callback,
    )
    headers = {"Authorization": f"Bearer {token}"}

    downloaded: list[Path] = []
    for item in _walk_drive(remote_path=remote_path, headers=headers):
        if len(downloaded) >= max_files:
            break
        name = item.get("name", "")
        suffix = Path(name).suffix.lower()
        if suffix not in extensions:
            continue
        download_url = item.get("@microsoft.graph.downloadUrl")
        if not download_url:
            continue
        local_path = destination_dir / _safe_relative_path(item)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        content = requests.get(download_url, timeout=300)
        content.raise_for_status()
        local_path.write_bytes(content.content)
        downloaded.append(local_path)
    return downloaded


def _get_token(
    *,
    client_id: str,
    tenant_id: str,
    token_cache_path: Path,
    device_flow_callback: Callable[[dict], None] | None,
) -> str:
    cache = msal.SerializableTokenCache()
    if token_cache_path.exists():
        cache.deserialize(token_cache_path.read_text())

    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        token_cache=cache,
    )

    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(DEFAULT_SCOPES, account=accounts[0])

    if not result:
        flow = app.initiate_device_flow(scopes=DEFAULT_SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(f"Failed to create device flow: {flow}")
        if device_flow_callback:
            device_flow_callback(flow)
        result = app.acquire_token_by_device_flow(flow)

    if cache.has_state_changed:
        token_cache_path.write_text(cache.serialize())

    if "access_token" not in result:
        raise RuntimeError(f"Failed to acquire token: {result}")
    return result["access_token"]


def _walk_drive(*, remote_path: str, headers: dict) -> Iterable[dict]:
    normalized = remote_path.strip("/")
    if normalized:
        url = f"{GRAPH_ROOT}/me/drive/root:/{normalized}:/children"
    else:
        url = f"{GRAPH_ROOT}/me/drive/root/children"

    while url:
        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("value", []):
            if "folder" in item:
                child_path = item.get("parentReference", {}).get("path", "")
                root_prefix = "/drive/root:"
                if root_prefix in child_path:
                    parent = child_path.split(root_prefix, 1)[1].strip("/")
                    next_path = f"{parent}/{item['name']}".strip("/")
                else:
                    next_path = item["name"]
                yield from _walk_drive(remote_path=next_path, headers=headers)
            else:
                yield item
        url = payload.get("@odata.nextLink")


def _safe_relative_path(item: dict) -> Path:
    parent_path = item.get("parentReference", {}).get("path", "")
    root_prefix = "/drive/root:"
    parent = ""
    if root_prefix in parent_path:
        parent = parent_path.split(root_prefix, 1)[1].strip("/")
    parts = [part for part in parent.split("/") if part and part not in {".", ".."}]
    return Path(*parts, item.get("name", "downloaded_file"))
