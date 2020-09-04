import logging
from logging import handlers
import os

def get_logger(config, save_dir, letter):
    mylogger = logging.getLogger(config.project_name)
    mylogger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s, %(message)s')

    if not os.path.exists(os.path.join(save_dir, letter)):
        os.makedirs(os.path.join(save_dir, letter))
    loghandler = handlers.TimedRotatingFileHandler(filename=os.path.join(save_dir, letter, 'logging.log'),
                                                   when='midnight',
                                                   interval=1,
                                                   encoding='utf-8')

    loghandler.setFormatter(formatter)
    loghandler.suffix = '%Y%m%d'
    mylogger.addHandler(loghandler)

    return mylogger