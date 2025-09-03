# %%
import torch
from torch import nn

from orquestra.qml.models.samplers.th import MultiDimGaussianSampler
from orquestra.qml.trainers import AdversarialTrainer

from utils import SmilesEncoding,SelfiesEncoding
from models.gan.molqcgan import MolGenerator, MolDiscriminator, MolQCGAN, sample_from_mol_logits
from orquestra.qml.data_loaders import new_data_loader
# %%
path_to_dataset = "data/KRAS_G12D/KRAS_G12D_inhibitors_update202209_updated_selfies.csv"
dataset_id = "insilico_KRAS"

# %%
n_epochs = 3
batch_size=512
# smiles = SmilesEncoding(
#     path_to_dataset,
#     dataset_identifier=dataset_id
# )

smiles = SelfiesEncoding(
        path_to_dataset,
    dataset_identifier=dataset_id
)


sequence_length = smiles.max_length
sample_dimension = 10
vocab_size = smiles.num_emd
padding_idx = smiles.pad_char_index

# %%
prior = MultiDimGaussianSampler((sequence_length, sample_dimension))

# NOTE: MolGenerator samples need to be passed through a softmax layer to get probabilities and then sampled from to get the actual encoded SMILES
generator = MolGenerator(
    noise_dim=sample_dimension,
    sequence_length=sequence_length,
    vocab_size=vocab_size,
    output_activation=torch.nn.Identity()  # we use identity because we want raw logits that get fed to discriminator
    
)
discriminator = MolDiscriminator(
    vocab_size=vocab_size,
    latent_dim=64,
    sequence_length=sequence_length,
    padding_index=padding_idx
)

mol_gan = MolQCGAN(generator, discriminator, prior)
# mol_gan.to_device(torch.device("mps:0"))

# %%
# data_loader = smiles.create_data_loader(batch_size=32)
encoded_samples_th = torch.tensor(smiles.encoded_samples)
data = encoded_samples_th.float()

data_loader = new_data_loader(
    data=data, batch_size=batch_size
).shuffle(12345)

data_loader.shuffle()

# %%
trainer = AdversarialTrainer()

# %%
train_cache = trainer.train(mol_gan, data_loader, n_epochs=n_epochs)

# %%
logit_samples = mol_gan.generate(10)
encoded_samples = sample_from_mol_logits(logit_samples)

# %%
print(smiles.decode_fn(encoded_samples))



