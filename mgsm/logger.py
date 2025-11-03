import json
import logging
from pathlib import Path


CURRENT_FOLDER = Path(__file__).parent
BASE_PATH = CURRENT_FOLDER / "logs"

def setup_logger(i):
    logger = logging.getLogger(f"result-{i}")
    logger.setLevel(logging.WARNING)

    # Create logs directory if it doesn't exist
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    

    # Create Handler: Output to the console
    log_path = BASE_PATH / f"result-{i}.log"
    file_handler = logging.FileHandler(log_path, mode='w') 
    file_handler.setLevel(logging.WARNING)

    # Create Formatter: define the log output format
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # Add Handler to logger
    logger.addHandler(file_handler)
    return logger


def log(logger, mode, input, output=None, mark=None):    
    if output:
        if mode == 'Init Team' or mode == 'Update Team':
            roles = json.dumps(output['roles'], indent=2)
            workflow = json.dumps(output['workflow'], indent=2)
            output = f'''# Roles:\n{roles}\n\n# Workflow:\n{workflow}'''

        logger.warning(f"==========={mode} Input===========\n{input}\n")
        logger.warning(f"==========={mode} Output===========\n{output}\n")  

    else:
        if mark == '-':
            logger.warning(f"-----------{mode}-----------\n{input}\n")  
        else:
            logger.warning(f"==========={mode}===========\n{input}\n")


def log_all(logger, logs):
    for item in logs:
        log(logger=logger, mode=item[0], input=item[1], output=item[2])
        