import flwr as fl

# Start Flower server
fl.server.start_server(
        server_address="172.31.7.102:8080",
    config=fl.server.ServerConfig(num_rounds=3), grpc_max_message_length=1024*1024*1024
)
