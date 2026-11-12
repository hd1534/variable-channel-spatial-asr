"""Authorize a personal Google account with a user-created desktop OAuth client.

Download credentials.json from Google Cloud first; see docs/personal-drive.md.
This helper never prints tokens and never uses a shared third-party OAuth app.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drive_api import (SCOPES, GoogleDriveTransport, StorageError, create_folder, execute,
                       private_json, validate_auth)


def authorize(client_secrets: Path, destination: Path, *, folder_name: str = "vcsasr-backups",
              reauthorize: bool = False) -> dict[str, Any]:
    """Perform a local loopback login and create or reconnect the app's backup folder."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    if destination.exists() and not reauthorize:
        raise StorageError("authorization already exists; use --reauthorize to change the authenticated account")
    try:
        client = json.loads(client_secrets.read_text())
    except (OSError, ValueError):
        raise StorageError("download the Desktop OAuth client JSON to runs/private/credentials.json first") from None
    if not isinstance(client, dict) or "installed" not in client:
        raise StorageError("the OAuth client must be a Desktop app, not a web app or service account")
    if not folder_name or "/" in folder_name or folder_name in (".", ".."):
        raise StorageError("choose one dedicated Drive folder name")
    flow = InstalledAppFlow.from_client_config(client, scopes=SCOPES)
    credentials = flow.run_local_server(
        host="localhost", port=0, access_type="offline", prompt="consent select_account",
        authorization_prompt_message="Complete personal-account authorization in the browser window.",
        success_message="Personal Drive authorization completed. You can close this window.",
        timeout_seconds=300,
    )
    if not credentials.refresh_token:
        raise StorageError("Google did not grant offline access; rerun authorization and accept the requested access")
    service = build("drive", "v3", credentials=credentials, cache_discovery=False, static_discovery=True)
    account = execute(service.about().get(fields="user(emailAddress)"))["user"]["emailAddress"]
    # Reuse only the folder selected by a previous setup for this account and this client.
    folder_id: str | None = None
    if destination.exists():
        previous = validate_auth(json.loads(destination.read_text()))
        if previous.get("account") == account and previous["credentials"]["client_id"] == credentials.client_id:
            folder_id = previous["folder_id"]
    if folder_id is None:
        folder_id = create_folder(service, folder_name)["id"]
    package = validate_auth({"schema": 1, "account": account, "folder_id": folder_id,
                             "credentials": json.loads(credentials.to_json())})
    GoogleDriveTransport(folder_id, lambda: service).prepare()
    private_json(destination, package)
    print(f"Personal Drive authorization saved privately: {destination}")
    print("Copy that JSON into the Colab Secret DRIVE_AUTH and enable notebook access. Do not paste it into chat.")
    print(f"Backup folder: https://drive.google.com/drive/folders/{folder_id}")
    return package


def main(argv: list[str] | None = None) -> int:
    """Run one explicit personal-account authorization on the user's local computer."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--client-secrets", type=Path, default=Path("runs/private/credentials.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/private/drive-auth.json"))
    parser.add_argument("--folder-name", default="vcsasr-backups")
    parser.add_argument("--reauthorize", action="store_true")
    args = parser.parse_args(argv)
    authorize(args.client_secrets, args.out, folder_name=args.folder_name, reauthorize=args.reauthorize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
