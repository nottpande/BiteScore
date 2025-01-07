import os
import sys
import logging

logging_format = "[%(acstime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir,"logging.log")
os.makedirs(log_filepath,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout) #put message in the terminal/stream
    ]
)

logger = logging.getLogger("logger")