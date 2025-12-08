from HumSpectra.sample import Sample, SampleCollection
import HumSpectra.fluorescence as fl
import HumSpectra.utilits as ut
import HumSpectra.ultraviolet as uv

from pathlib import Path
import numpy as np
import sys
import pandas as pd

project_root = Path.cwd()
sys.path.insert(0, str(project_root))
from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, PARAFAC_PATH,STAT_PATH

date_list = ["2024","2025"]
date_folder_list = [INIT_DATA_2024,INIT_DATA_2025]

collection = SampleCollection(cache_dir=Path.joinpath(Path.cwd(),"Initial_data","data_cache"))

parafac_data = pd.read_csv(Path.joinpath(PARAFAC_PATH,"data","parafac.csv"))

for i, folder in enumerate(date_folder_list):
    
    fluo_path = Path.joinpath(folder, "Fluorescence")
    uv_path = Path.joinpath(folder, "UV","Individual_spectra")
    
    carbon_data = pd.read_excel(Path.joinpath(folder, 'C_org.xlsx'))
    carbon_data.Sample = carbon_data.Sample.apply(lambda x: x+f'-{date_list[i]}')
    
    geodata = pd.read_excel(Path.joinpath(folder, 'GeoData.xlsx'))
    geodata.Sample = geodata.Sample.apply(lambda x: x+f'-{date_list[i]}')
    
    fluo_spectra_list = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
    
    uv_abs_spectra_list = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")
    uv_reflect_spectra_list = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="reflection")
    
    for fl_spectra in fluo_spectra_list:
        
        sample_name = fl_spectra.attrs['name']
        
        if sample_name not in collection.samples:
        
            sample = collection.create_sample_from_data(
            sample_id = sample_name,
            sample_subclass = fl_spectra.attrs['subclass'],
            sample_class = fl_spectra.attrs['class'],
            fluorescence_eem = fl_spectra
                )
        
        else:
            
            sample = collection.samples[sample_name]
            sample.fluorescence_eem = fl_spectra
            
        sample.add_tag("raw_fluo")
        
        fl_spectra_cutted = fl.cut_spectra(fl_spectra,255,450,255,600)
        sample.cutted_fluo_eem = fl_spectra_cutted
        sample.add_tag("cutted_fluo")
        
        sample.descriptors['asm_280'] = fl.calc_asm_280(fl_spectra)
        sample.descriptors['asm_350'] = fl.calc_asm_350(fl_spectra)
        sample.descriptors['HIX'] = fl.calc_humin_index(fl_spectra)
        sample.descriptors['FIX'] = fl.calc_fluo_index(fl_spectra)
                          
    for uv_abs in uv_abs_spectra_list:
        
        sample_name = uv_abs.attrs['name']
        
        if sample_name not in collection.samples:
        
            sample = collection.create_sample_from_data(
            sample_id = sample_name,
            sample_subclass = uv_abs.attrs['subclass'],
            sample_class = uv_abs.attrs['class'],
            uv_vis_absorption = uv_abs
                )
        
        else:
            
            sample = collection.samples[sample_name]
            sample.uv_vis_absorption = uv_abs
            
        sample.add_tag("raw_uv_abs")
        
        if uv_abs.min().min() < 0:
            uv_abs_recall = uv_abs - uv_abs.min() + 0.00001
        
        else:
            uv_abs_recall = uv_abs
        
        sample.uv_vis_absorption = uv_abs_recall    
        sample.add_tag("uv_abs_baseline")
        
        smooth_uv_abs = uv.smooth_uv_spectrum(uv_abs_recall, window_size=9, threshold_factor=1.5)
        smooth_uv_abs = smooth_uv_abs.iloc[30:]
        
        sample.uv_vis_absorption_smooth = smooth_uv_abs
        sample.add_tag("smooth_uv_abs")   
        
        sample.descriptors['B1'] = uv.calc_cdom_descriptors(uv_abs_recall)['B1_prime']
        sample.descriptors['B2'] = uv.calc_cdom_descriptors(uv_abs_recall)['B2_prime']
        sample.descriptors['ag_275'] = uv.calc_cdom_descriptors(uv_abs_recall)['ag_275']
        sample.descriptors['ag_380'] = uv.calc_cdom_descriptors(uv_abs_recall)['ag_380']
        sample.descriptors['S_380_443'] = uv.calc_cdom_descriptors(uv_abs_recall)['S_380_443']
        sample.descriptors['B1'] = uv.calc_cdom_descriptors(uv_abs_recall)['B1_prime']                  
        sample.descriptors['E2E3'] = uv.calc_e2_e3(uv_abs_recall)
        sample.descriptors['E3E4'] = uv.calc_e3_e4(uv_abs_recall)
        sample.descriptors['E4E6'] = uv.calc_e4_e6(uv_abs_recall)                              
        sample.descriptors['lambda'] = uv.calc_lambda_UV(uv_abs_recall)
                    
    for uv_refl in uv_reflect_spectra_list:
        
        sample_name = uv_refl.attrs['name'].replace("-R","")
        
        if sample_name not in collection.samples:
        
            sample = collection.create_sample_from_data(
            sample_id = sample_name,
            sample_subclass = uv_refl.attrs['subclass'],
            sample_class = uv_refl.attrs['class'],
            uv_vis_reflection = uv_refl
                )
        
        else:
            
            sample = collection.samples[sample_name]
            sample.uv_vis_reflection = uv_refl
            
        sample.add_tag("raw_uv_refl")
        
        smooth_uv_refl = uv.smooth_uv_spectrum(uv_refl, window_size=9, threshold_factor=1.5)
        smooth_uv_refl = smooth_uv_refl.iloc[20:]
        
        sample.uv_vis_reflection_smooth = smooth_uv_refl
        sample.add_tag("smooth_uv_refl")
        
        sample.descriptors['r280_460'] = uv.calc_r280_460(uv_refl)
        sample.descriptors['r412_547'] = uv.calc_r412_547(uv_refl)
        sample.descriptors['r412_670'] = uv.calc_r412_670(uv_refl)
        sample.descriptors['r460_560'] = uv.calc_r460_560(uv_refl)
        sample.descriptors['r490_555'] = uv.calc_r490_555(uv_refl)
        sample.descriptors['ir_19_20'] = uv.calc_ir_19_20(uv_refl)
        sample.descriptors['ir_2_18'] = uv.calc_ir_2_18(uv_refl)
        sample.descriptors['ir_4_5'] = uv.calc_ir_4_5(uv_refl)
        sample.descriptors['ir_7_8'] = uv.calc_ir_7_8(uv_refl)
                          
    for k,row in carbon_data.iterrows():
        
        sample_name = row['Sample']
        
        if sample_name not in collection.samples:
            
            #print(f"Нет спектральных данных образца {sample_name}")
            continue
        
        else:
            
            sample = collection.samples[sample_name]
            sample.org_carbon = row["C_org"]
            
            if sample.org_carbon is None:
                
                raise ValueError("Значение TOC неверно")
            
        sample.add_tag("org_carbon")
        
        if sample.uv_vis_absorption is not None:
            sample.descriptors['Suva'] = uv.calc_suva(sample.uv_vis_absorption,toc=sample.org_carbon,debug=True)  
                  
    for k,row in geodata.iterrows():
        
        sample_name = row['Sample']
        
        if sample_name not in collection.samples:
        
            #print(f"Нет спектральных данных образца {sample_name}")
            continue
        
        else:
            
            sample = collection.samples[sample_name]
            sample.pH = row["pH"]
            sample.Eh = row['eH']
            sample.latitude = row['latitude']
            sample.lontitude = row['lontitude']
            
        sample.add_tag("geo")
            
for k,row in parafac_data.iterrows():
    
    sample_name = row['Sample']
    
    if sample_name not in collection.samples:
    
        print(f"Нет спектральных данных образца {sample_name}")
        continue
    
    else:
        
        sample = collection.samples[sample_name]
        
        for column in parafac_data.columns:
            
            sample.descriptors[column] = row[column]
    
    sample.add_tag("parafac")
    
print(f"Всего образцов в коллекции: {len(collection.samples)}")
collection.save_collection("KNP_samples.pkl")

data = collection.create_flexible_descriptors_table(sample_attributes=['org_carbon','sample_subclass','sample_class'],min_coverage=0)
data.dropna(inplace=True)
data.to_excel(Path.joinpath(STAT_PATH,"descriptors.xlsx"))