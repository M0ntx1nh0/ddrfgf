import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False


def get_app_mode() -> str:
    load_dotenv()
    return os.getenv("APP_MODE", "full").strip().lower()


def is_home_only_mode() -> bool:
    return get_app_mode() == "home_only"
