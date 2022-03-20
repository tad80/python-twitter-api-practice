import logging
from logging import getLogger,config,handlers
from pythonjsonlogger import jsonlogger


class Logger:
    """
    Custom logger class
    """

    def __init__(self, configfile, batch_name):
        """
        Init constructor
        """
        config.fileConfig(configfile, disable_existing_loggers=False)
        rootlogger = logging.getLogger()
        self.log = rootlogger.getChild(batch_name)


    def addLogger(name):
        """
        If you want to add another logger
        """
        additionalLogger = logging.getLogger(name)
        additionalLogger.setLevel(logging.INFO)

