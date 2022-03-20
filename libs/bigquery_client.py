from configparser import ConfigParser
from google.cloud import bigquery
from retry import retry


class BigQueryClient:
    """
    BigQuery client class
    """


    def __init__(self, config):
        """
        Init constructor
        """
        self.parser = ConfigParser(interpolation=None)
        self.parser.read(config)
        self.project_id = self.parser["BigQuery"]["project_id"]
        self.client = bigquery.Client()
        self.job_config = bigquery.LoadJobConfig(
            autodetect=True,
            schema_update_options=bigquery.job.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        )


    def load(self, json, dataset_id, table_id):
        """
        Wrapper of bigquery.load_table_from_json.
        Returns number of rows after the load.
        """
        destination = ".".join([self.project_id, dataset_id, table_id])
        self.client.load_table_from_json(
            json, destination, job_config=self.job_config
        ).result()
        return self.count(destination)

    @retry(delay=1, tries=3)
    def count(self, destination):
        return self.client.get_table(destination).num_rows
