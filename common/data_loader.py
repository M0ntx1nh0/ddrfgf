import io
import json
import os

import pandas as pd
import requests
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False


DEFAULT_LOCAL_CSV = "data/Jugadoresv1.csv"


def _build_drive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _read_csv_from_bytes(content: bytes) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(content), sep=";", encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(content), sep=";")


def _download_drive_file_with_service_account(
    file_id: str,
    service_account_file: str,
    service_account_json: str,
) -> bytes:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = None

    if service_account_file:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=scopes,
        )
    elif service_account_json:
        credentials_info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=scopes,
        )
    else:
        raise ValueError("Missing service account credentials")

    drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    request = drive_service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return buffer.getvalue()


def load_players_data() -> pd.DataFrame:
    """
    Carga el CSV principal de jugadores desde:
    1) Google Drive privado con service account
    2) URL directa en WYSCOUT_CSV_URL
    3) Google Drive publico con WYSCOUT_DRIVE_FILE_ID
    4) Ruta local en WYSCOUT_CSV_LOCAL_PATH (fallback)
    """
    load_dotenv()

    csv_url = os.getenv("WYSCOUT_CSV_URL", "").strip()
    drive_file_id = os.getenv("WYSCOUT_DRIVE_FILE_ID", "").strip()
    local_path = os.getenv("WYSCOUT_CSV_LOCAL_PATH", DEFAULT_LOCAL_CSV).strip()
    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    try:
        if drive_file_id and (service_account_file or service_account_json):
            csv_bytes = _download_drive_file_with_service_account(
                file_id=drive_file_id,
                service_account_file=service_account_file,
                service_account_json=service_account_json,
            )
            return _read_csv_from_bytes(csv_bytes)

        if csv_url:
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            return _read_csv_from_bytes(response.content)

        if drive_file_id:
            drive_url = _build_drive_download_url(drive_file_id)
            response = requests.get(drive_url, timeout=30)
            response.raise_for_status()
            return _read_csv_from_bytes(response.content)
    except Exception:
        if os.path.exists(local_path):
            return pd.read_csv(local_path, sep=";", encoding="utf-8")
        raise

    return pd.read_csv(local_path, sep=";", encoding="utf-8")
