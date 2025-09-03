import torch
import cloudpickle 
import tqdm
from functools import partial
import time
from syba.syba import SybaClassifier

from utils import (
    compute_compound_stats,
    generate_bulk_samples,
)
from utils.filter import (
    apply_filters,
    filter_phosphorus,
    substructure_violations,
    maximum_ring_size,
    # lipinski_filter,
    get_diversity,
    passes_wehi_mcf,
    pains_filt,
)
from rdkit import Chem
from rdkit.Chem import Draw
from datetime import datetime
diversity_fn = get_diversity
start_time = time.time()
syba = SybaClassifier()
syba.fitDefaultScore()
print("Syba fitting time: ", time.time() - start_time)
from utils.data import compund_to_csv
def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = cloudpickle.load(f)
    return obj


def combine_filter(smiles_compound, max_mol_weight=800):
    pass_all = []
    i = 0

    with tqdm.tqdm(total=len(smiles_compound)) as pbar:
        for smile_ in smiles_compound:
            pbar.set_description(f"Filtered {i} / {len(smiles_compound)}.")
            try:
                if (
                    apply_filters(smile_, max_mol_weight)
                    and smile_ not in pass_all
                    and (syba.predict(smile_) > 0)
                    and passes_wehi_mcf(smile_)
                    and (len(pains_filt(Chem.MolFromSmiles(smile_))) == 0)
                ):

                    pass_all.append(smile_)
            except:
                pass
            i += 1
            pbar.update()
    return pass_all

n_samples = 10000
max_mol_weight = 600
path ="/home/mghazi/workspace/insilico-drug-discovery/experiment_results/saved_models"
saved_compounds_dir ="/home/mghazi/workspace/insilico-drug-discovery/experiment_results/samples"
run_date_time = datetime.today().strftime("%Y_%d_%mT%H_%M_%S.%f")
file_path = f"{path}/noisy-lstm-v3-noisy-lstm-v3-2023_22_02T19_21_13.397418-22Feb2023T1921.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new= load_obj(file_path)
model = new['model']
model.to_device(device)
selfies = new['selfies']
prior = new['prior']

validity_fn = partial(combine_filter, max_mol_weight=max_mol_weight) 

train_compounds = selfies.train_samples
decoder_fn = selfies.decode_fn


# samples_ = torch.tensor(prior.generate(10))
# g_samples = selfies.decode_fn(model.generate(samples_.float()))
for i in range(1,1000):
    g_samples = generate_bulk_samples(
            model,
            n_samples,
            5000,
            2,
            prior=prior,
            verbose=True,
            unique=True,
        )


    results_analysis = compute_compound_stats(
                    g_samples, decoder_fn, get_diversity, validity_fn, train_compounds
                )
    compund_to_csv(
                    results_analysis, file_path=f"{saved_compounds_dir}/{i}_valid_unique_samples{run_date_time}.csv"
                )
    print(f"{i},diversity_fraction={results_analysis.diversity_fraction}, filter_fraction={results_analysis.filter_fraction}, unique_fraction={results_analysis.unique_fraction}")