import argparse
from configparser import ConfigParser
from datetime import datetime
import time
from libs.logger import Logger
from libs.bigquery_client import BigQueryClient
from libs.twitter_v2_client import TwitterV2Client


class TweetsCollectorService:
    """
    Fetches tweets from Twitter v2 API and injests them to BigQuery
    """


    def __init__(self, twitter_config, bigquery_config, start, end):
        """
        Init constructor
        """
        self.logger = Logger("./config/logger.ini", self.__class__.__name__)
        self.parser = ConfigParser(interpolation=None)
        self.parser.read(twitter_config)
        self.bq_config = ConfigParser(interpolation=None)
        self.bq_config.read(bigquery_config)
        start_time = datetime.strftime(datetime.strptime(start, "%Y-%m-%d"), "%Y-%m-%dT%H:%M:%SZ")
        end_time = datetime.strftime(datetime.strptime(end, "%Y-%m-%d"), "%Y-%m-%dT%H:%M:%SZ")
        self.twitter_params = {
            'max_results': self.parser["TwitterAPI"]["max_results"],
            'query': self.parser["TwitterAPI"]["query"],
            'start_time': start_time,
            'end_time': end_time,
            'expansions': self.parser["TwitterAPI"]["expansions"],
            'tweet.fields': self.parser["TwitterAPI"]["tweet_fields"],
            'user.fields': self.parser["TwitterAPI"]["user_fields"],
            'place.fields': self.parser["TwitterAPI"]["place_fields"],
            'media.fields': self.parser["TwitterAPI"]["media_fields"]
        }
        self.logger.log.info("loaded config %s" % self.twitter_params)
        self.twitter = TwitterV2Client(twitter_config)
        self.bq = BigQueryClient(bigquery_config)


    def main(self):
        """
        Main method
        """
        go_ahead = True
        while go_ahead is True:
            response = self.twitter.call(self.parser["TwitterUrl"]["search"], self.twitter_params)
            self.logger.log.debug(response["data"])
            rows = self.bq.load(response["data"], self.bq_config["BigQuery"]["dataset_id"], self.bq_config["BigQuery"]["table_id"])
            self.logger.log.info("%s rows added to BigQuery" % rows)
            if response["meta"]["next_token"] is None:
                go_ahead = False
            else:
                self.twitter_params["next_token"] = response["meta"]["next_token"]
            time.sleep(3)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Fetches tweets from Twitter v2 API and injests them to BigQuery.")
    PARSER.add_argument("--twitter_config", required=True, help="Twitter v2 API config file.")
    PARSER.add_argument("--bigquery_config", required=True, help="BigQuery config file.")
    PARSER.add_argument("--start", required=True, help="Start date. YYYY-MM-DD")
    PARSER.add_argument("--end", required=True, help="End date. YYYY-MM-DD")
    ARGS = PARSER.parse_args()
    TweetsCollectorService(ARGS.twitter_config, ARGS.bigquery_config, ARGS.start, ARGS.end).main()
