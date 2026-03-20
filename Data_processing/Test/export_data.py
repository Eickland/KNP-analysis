import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import HumSpectra.ultraviolet as uv
from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import CACHE_DATA_PATH, PROJECT_ROOT

print(PROJECT_ROOT)

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

for name, sample in collection.samples.items():
    
    if sample.fluorescence_eem is not None:
        sample.fluorescence_eem.to_csv(fr"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Initial_data\Fluorescence\{sample.sample_id}.csv")
        
    if sample.uv_vis_absorption is not None:

        sample.uv_vis_absorption.to_csv(fr"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Initial_data\UV\absorption\{sample.sample_id}.csv")
        
    if sample.uv_vis_reflection is not None:

        sample.uv_vis_reflection.to_csv(fr"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Initial_data\UV\reflect\{sample.sample_id}.csv")