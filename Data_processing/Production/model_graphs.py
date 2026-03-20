import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import numpy as np

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import CACHE_DATA_PATH

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

# Настройки стиля
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

model_data = pd.read_excel(Path.joinpath(Path.cwd(),"Server","Received_data","Model",f"Model_accuracy_cross-val.xlsx"))

collection.visualize_tag_communities(resolution=0.8)
plt.show()
