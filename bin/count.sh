#!/bin/bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/tadashi.a.nakamura/workspace/twitter-practice/config/google_service_account.json"
pipenv run python tweets_counter_service.py --twitter_config config/twitter_api_count.ini --bigquery_config config/bigquery_tweets_count.ini --start 2011-01-01 --end 2022-08-27
