from rdkit import Chem
import sys
from pathlib import Path
SELFIES_coder_path = Path("../SELFIES_coder")
sys.path.append(SELFIES_coder_path.as_posix())
import SELFIES_coder as SELFIES_coder
import pandas as pd

def read_and_prepare_data(file, reps):
    #read file
    data = pd.read_excel(file)
    #read SMILES
    #data = data[data['SELFIES_length'] < 55]
    data = data['SMILES']
    #prepare rdkit representations
    mols = [Chem.MolFromSmiles(smi) for smi in data]

    smiles_training = []
    #prepare various representations for each of molecule
    for i in range(reps):
        for smi in mols:
            try:
                new_smi = Chem.MolToSmiles(smi, doRandom=True)
                smiles_training.append(new_smi)
            except:
                print("Error when processing", i, smi)
    
    selfies_training = [SELFIES_coder.SMILES_to_SELFIES(smi) for smi in smiles_training]
    drop_duplicates_self_train = list(set(selfies_training))

    to_be_tested__ = [SELFIES_coder.SELFIES_to_SMILES(self) for self in drop_duplicates_self_train]
    drop_duplicates_smmi_train = list(set(to_be_tested__))
    print("The number of training data is "+str(len(drop_duplicates_smmi_train)))

    training_smiles = pd.DataFrame(data=drop_duplicates_smmi_train, columns=['SMILES'])

    return training_smiles.to_parquet('../Data/training_smiles.parquet')


if __name__ == "__main__":

    read_and_prepare_data('../Data/Kolchicyna_prepared_data.xlsx', 1000)

