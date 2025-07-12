import logging, sys
from logging import StreamHandler

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '[36m', 'INFO': '[32m', 'WARNING': '[33m',
        'ERROR': '[31m', 'CRITICAL': '[35m'
    }
    RESET = '[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[StreamHandler(sys.stdout)]
    )
