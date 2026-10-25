"""Dataset defaults shared by preparation and training entrypoints."""

PEMS_DATASETS = {
    "pems04": {"num_nodes": 307, "raw_data": "data/PEMS04/PEMS04.npz",
               "distance_csv": "data/PEMS04/PEMS04.csv", "past_source": "raw"},
    "pems07": {"num_nodes": 883, "raw_data": "data/PEMS07/PEMS07.npz",
               "distance_csv": "data/PEMS07/PEMS07.csv", "past_source": "time"},
    "pems08": {"num_nodes": 170, "raw_data": "data/PEMS08/PEMS08.npz",
               "distance_csv": "data/PEMS08/PEMS08.csv", "past_source": "raw"},
}
