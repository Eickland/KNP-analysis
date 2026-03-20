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

for name, sample in collection.samples.items():

    if sample.sample_class == 'ADOM':
        sample.add_tag("ADOM")
        
        if sample.sample_subclass == 'Storage 1':
            sample.add_tag('S1')
            
        if sample.sample_subclass == 'Storage 2':
            sample.add_tag('S2')
            
        if sample.sample_subclass == 'Storage 3':
            sample.add_tag('S3')
            
        if sample.sample_subclass == 'Storage 4':
            sample.add_tag('S4')
            
        if sample.sample_subclass == 'Storage 5':
            sample.add_tag('S5')
            
        if sample.sample_subclass == 'Storage 6':
            sample.add_tag('S6')
            
        if sample.sample_subclass == 'Storage 7':
            sample.add_tag('S7')
            
        if sample.sample_subclass == 'Storage 7':
            sample.add_tag('S7')
            
        if sample.sample_subclass == 'Storage 8':
            sample.add_tag('S8')
            
        if sample.sample_subclass == 'Storage 9':
            sample.add_tag('S9')
            
        if sample.sample_subclass == 'Storage 10':
            sample.add_tag('S10')
            
    elif sample.sample_class == 'Coal':
        sample.add_tag('Coal')
        sample.add_tag('HS')
        
        if sample.sample_subclass == 'CHA':
            sample.add_tag('CHA')
            sample.add_tag('HA')
            
        elif sample.sample_subclass == 'CHF':
            sample.add_tag('CHF')
            sample.add_tag('HF')
            
        elif sample.sample_class == 'CFA':
            sample.add_tag('CFA')
            sample.add_tag('FA')
            
    elif sample.sample_class == 'Peat':
        sample.add_tag('Peat')
        sample.add_tag('HS')
        
        if sample.sample_subclass == 'PHA':
            sample.add_tag('PHA')
            sample.add_tag('HA')
            
        elif sample.sample_subclass == 'PHF':
            sample.add_tag('PHF')
            sample.add_tag('HF')
            
        elif sample.sample_class == 'PFA':
            sample.add_tag('PFA')
            sample.add_tag('FA')
            
    elif sample.sample_class == 'Soil':
        sample.add_tag('Soil')
        sample.add_tag('HS')
        
        if sample.sample_subclass == 'SHA':
            sample.add_tag('SHA')
            sample.add_tag('HA')
            
        elif sample.sample_subclass == 'SHF':
            sample.add_tag('SHF')
            sample.add_tag('HF')
            
        elif sample.sample_class == 'SFA':
            sample.add_tag('SFA')
            sample.add_tag('FA')
            
    elif sample.sample_class == 'Lg':
        sample.add_tag('Lg')
        
    elif sample.sample_class == 'Lst':
        sample.add_tag('Lg')
        sample.add_tag('Lst')
        
collection.save_collection("KNP_samples.pkl")    
        
        