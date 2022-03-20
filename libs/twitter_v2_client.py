from configparser import ConfigParser
from logging import getLogger
import requests
from libs.logger import Logger


class TwitterV2Client:
    """
    Twitter API v2 client Class
    """


    def __init__(self, config):
        """
        Init constructor
        """
        self.logger = Logger("./config/logger.ini", self.__class__.__name__)
        self.parser = ConfigParser(interpolation=None)
        self.parser.read(config)
        self.bearer_token = self.parser["TwitterAuth"]["bearer_token"]


    def __bearer_oauth(self, request):
        """
        Method required by bearer token authentication
        """
        request.headers["Authorization"] = f"Bearer {self.bearer_token}"
        request.headers["User-Agent"] = self.__class__.__name__
        return request


    def call(self, url, params):
        """
        Makes call against Twtiter API v2
        """
        self.logger.log.info("calling %s" % url)
        response = requests.request("GET", url, auth=self.__bearer_oauth, params=params)
        self.logger.log.info(response.headers)
        if response.status_code == 200:
            self.logger.log.info("%s %s" % (response.status_code, response.json()["meta"]))
        else:
            self.logger.log.info("%s %s %s" % (response.status_code, response.json()["title"], response.json()["detail"]))
            raise Exception(response.status_code, response.text)
        return response.json()
