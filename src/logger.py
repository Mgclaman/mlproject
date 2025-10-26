import logging
import os
from datetime import datetime 

# 1. Define the unique log file name (e.g., 2025-10-26_13-26-51.log)
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# 2. Define the path for the log *directory* ('logs' folder at the root of the project)
LOG_DIR = os.path.join(os.getcwd(), "logs")

# 3. Create the 'logs' directory if it doesn't exist
# This ensures the folder is there before the logging system tries to write the file.
os.makedirs(LOG_DIR, exist_ok=True)

# 4. Define the complete, final path for the log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# 5. Configure the root logger to write to the new file
logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILE_PATH,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(message)s",
)

if __name__ == "__main__":
    # This message will create the file and be the first line written to it
    logging.info("Logging has started")