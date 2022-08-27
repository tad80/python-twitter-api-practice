#!/bin/bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/tadashi.a.nakamura/workspace/twitter-practice/config/google_service_account.json"
pipenv run python tweets_collector_service.py --twitter_config config/twitter_api_search.ini --bigquery_config config/bigquery.ini --start 2012-01-01 --end 2012-12-31
