from app.cli import parse
from app.app_runner import AppRunner

if __name__ == "__main__":
    cfg = parse()
    AppRunner(cfg).run()
