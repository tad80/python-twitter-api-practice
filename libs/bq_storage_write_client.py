from configparser import ConfigParser
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1
from google.cloud.bigquery_storage_v1 import types
from google.cloud.bigquery_storage_v1 import writer
from google.protobuf import descriptor_pb2
from retry import retry


class BqStorageWriteApiClient:
    """
    BigQuery Storage Write API Client class
    """


    def __init__(self, config):
        """
        Init constructor
        """
        self.parser = ConfigParser(interpolation=None)
        self.parser.read(config)
        self.project_id = self.parser["BigQuery"]["project_id"]
        self.client = bigquery.Client()
        self.table = self.client.get_table(self.parser["BigQuery"]["table_id"])


    @retry(delay=5, tries=3)
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


    def append_rows_pending(project_id: str, dataset_id: str, table_id: str):

        """Create a write stream, write some sample data, and commit the stream."""
        write_client = bigquery_storage_v1.BigQueryWriteClient()
        parent = write_client.table_path(project_id, dataset_id, table_id)
        write_stream = types.WriteStream()

        # When creating the stream, choose the type. Use the PENDING type to wait
        # until the stream is committed before it is visible. See:
        # https://cloud.google.com/bigquery/docs/reference/storage/rpc/google.cloud.bigquery.storage.v1#google.cloud.bigquery.storage.v1.WriteStream.Type
        write_stream.type_ = types.WriteStream.Type.PENDING
        write_stream = write_client.create_write_stream(
            parent=parent, write_stream=write_stream
        )
        stream_name = write_stream.name

        # Create a template with fields needed for the first request.
        request_template = types.AppendRowsRequest()

        # The initial request must contain the stream name.
        request_template.write_stream = stream_name

        # So that BigQuery knows how to parse the serialized_rows, generate a
        # protocol buffer representation of your message descriptor.
        proto_schema = types.ProtoSchema()
        proto_descriptor = descriptor_pb2.DescriptorProto()
        customer_record_pb2.CustomerRecord.DESCRIPTOR.CopyToProto(proto_descriptor)
        proto_schema.proto_descriptor = proto_descriptor
        proto_data = types.AppendRowsRequest.ProtoData()
        proto_data.writer_schema = proto_schema
        request_template.proto_rows = proto_data

        # Some stream types support an unbounded number of requests. Construct an
        # AppendRowsStream to send an arbitrary number of requests to a stream.
        append_rows_stream = writer.AppendRowsStream(write_client, request_template)

        # Create a batch of row data by appending proto2 serialized bytes to the
        # serialized_rows repeated field.
        proto_rows = types.ProtoRows()
        proto_rows.serialized_rows.append(create_row_data(1, "Alice"))
        proto_rows.serialized_rows.append(create_row_data(2, "Bob"))

        # Set an offset to allow resuming this stream if the connection breaks.
        # Keep track of which requests the server has acknowledged and resume the
        # stream at the first non-acknowledged message. If the server has already
        # processed a message with that offset, it will return an ALREADY_EXISTS
        # error, which can be safely ignored.
        #
        # The first request must always have an offset of 0.
        request = types.AppendRowsRequest()
        request.offset = 0
        proto_data = types.AppendRowsRequest.ProtoData()
        proto_data.rows = proto_rows
        request.proto_rows = proto_data

        response_future_1 = append_rows_stream.send(request)

        # Send another batch.
        proto_rows = types.ProtoRows()
        proto_rows.serialized_rows.append(create_row_data(3, "Charles"))

        # Since this is the second request, you only need to include the row data.
        # The name of the stream and protocol buffers DESCRIPTOR is only needed in
        # the first request.
        request = types.AppendRowsRequest()
        proto_data = types.AppendRowsRequest.ProtoData()
        proto_data.rows = proto_rows
        request.proto_rows = proto_data

        # Offset must equal the number of rows that were previously sent.
        request.offset = 2

        response_future_2 = append_rows_stream.send(request)

        print(response_future_1.result())
        print(response_future_2.result())

        # Shutdown background threads and close the streaming connection.
        append_rows_stream.close()

        # A PENDING type stream must be "finalized" before being committed. No new
        # records can be written to the stream after this method has been called.
        write_client.finalize_write_stream(name=write_stream.name)

        # Commit the stream you created earlier.
        batch_commit_write_streams_request = types.BatchCommitWriteStreamsRequest()
        batch_commit_write_streams_request.parent = parent
        batch_commit_write_streams_request.write_streams = [write_stream.name]
        write_client.batch_commit_write_streams(batch_commit_write_streams_request)

        print(f"Writes to stream: '{write_stream.name}' have been committed.")
