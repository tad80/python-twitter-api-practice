# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

CMD exec GOOGLE_APPLICATION_CREDENTIALS="/app/config/google_service_account.json" python tweets_collector_service.py --twitter_config config/twitter_api.ini --bigquery_config config/bigquery.ini --start 2021-12-11 --end 2021-12-21
