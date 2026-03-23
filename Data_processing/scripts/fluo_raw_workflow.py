from HumSpectra.sample import Sample, SampleCollection
import HumSpectra.fluorescence as fl
import HumSpectra.utilits as ut
import HumSpectra.ultraviolet as uv

from pathlib import Path
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

raw_data_dir = Path(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\raw_data")
result_data_dir = Path(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\result_data")

fluo_spectra_list = ut.load_spectra_data(
    raw_data_dir, 
    fl.read_fluo_3d,
    assign_class_type=False
)

for fl_spectra in fluo_spectra_list:
    fl_spectra = fl.cut_spectra(fl_spectra,255,450,255,600)
    fl.plot_heat_map(fl_spectra)
    plt.savefig(fr"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\plots\{fl_spectra.attrs['name']}.png")

fl_data = ut.spectra_to_df(fluo_spectra_list, {
    "Asm_350": fl.calc_asm_350, "Asm_280": fl.calc_asm_280,
    "FIX": fl.calc_fluo_index, "HIX": fl.calc_humin_index
}, class_filter=False, check_class_type=False)

fl_data.to_csv(str(result_data_dir)+fr"\result.csv",index=False)