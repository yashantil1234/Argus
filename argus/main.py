import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


def main() -> None:
    app_name = os.getenv("APP_NAME", "Argus")
    app_env = os.getenv("APP_ENV", "development")
    print(f"Starting {app_name} in {app_env} mode...")
    print("Argus autonomous research environment initialized successfully.")


if __name__ == "__main__":
    main()
