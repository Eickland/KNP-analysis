from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

INIT_DATA = PROJECT_ROOT / 'Initial_data'

INIT_DATA_2024 = PROJECT_ROOT / 'Initial_data' / '2024'
INIT_DATA_2025 = PROJECT_ROOT / 'Initial_data' / '2025'
INIT_DATA = PROJECT_ROOT / 'Initial_data'


RECEIVED_DATA = PROJECT_ROOT / 'Server' / 'Received_data'

CLASSIFICATION_PATH = PROJECT_ROOT / 'Server' / 'Received_data' /'Model'/ 'Classification'
CLUSTERING_PATH = PROJECT_ROOT / 'Server' / 'Received_data'/'Model' / 'Clustering'

PARAFAC_PATH = PROJECT_ROOT  / 'Received_data' / 'Parafac'
STAT_PATH = PROJECT_ROOT / 'Server' / 'Received_data' / 'Statistics'
SPECTRA_PATH = PROJECT_ROOT / 'Server' / 'Received_data' / 'Spectra'

CACHE_DATA_PATH = PROJECT_ROOT / 'Initial_data' / '_data_cache'
