import logging
import os
import datetime
import sys

LOG_FILE = f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
    handlers = [
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)