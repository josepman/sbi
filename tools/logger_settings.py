def init_logger(output_path, name, lg_msg):
    import logging

    _log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"

    logger = logging.getLogger(lg_msg)
    hdlr = logging.FileHandler(f'{output_path}/{name}')     # Handler specifies the destination of log messages
    formatter = logging.Formatter(_log_format)  # Formatter specifies the structure of the log message
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

    return logger


def predef_logger():
    import sys
    import logging
    from logging.config import dictConfig
    
    logging_config = dict(
        version=1,
        formatters={
            'verbose': {
                'format': ("[%(asctime)s] %(levelname)s "
                           "[%(name)s:%(lineno)s] %(message)s"),
                'datefmt': "%d/%b/%Y %H:%M:%S",
            },
            'simple': {
                'format': '%(levelname)s %(message)s',
            },
        },
        handlers={
            'api-logger': {'class': 'logging.handlers.RotatingFileHandler',
                               'formatter': 'verbose',
                               'level': logging.DEBUG,
                               'filename': 'logs.log',
                               'maxBytes': 52428800,
                               'backupCount': 7},
            'batch-process-logger': {'class': 'logging.handlers.RotatingFileHandler',
                                 'formatter': 'verbose',
                                 'level': logging.DEBUG,
                                 'filename': 'batch.log',
                                 'maxBytes': 52428800,
                                 'backupCount': 7},
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'simple',
                'stream': sys.stdout,
            },
        },
        loggers={
            'api_logger': {
                'handlers': ['api-logger', 'console'],
                'level': logging.DEBUG
            },
            'batch_process_logger': {
                'handlers': ['batch-process-logger', 'console'],
                'level': logging.DEBUG
            }
        }
    )
    
    dictConfig(logging_config)
    
    api_logger = logging.getLogger('api_logger')
    batch_process_logger = logging.getLogger('batch_process_logger')
    
    return api_logger, batch_process_logger