#Libraries import
import pandas as pd
import numpy as np
from mordred import Calculator, descriptors
import mordred
from rdkit import Chem
from syba.syba import SybaClassifier
from rdkit.Chem import PandasTools
import joblib

import sys
from pathlib import Path
prediction_mode_path = Path("../module")
sys.path.append(prediction_mode_path.as_posix())
import models_creation as pred_model

def is_morder_missing(x):
    return np.nan if type(x) == mordred.error.Missing or type(x) == mordred.error.Error else x 

if __name__ == "__main__":
    #load initial structures
    initial_structures = pd.read_excel('../Data/Kolchicyna_prepared_data.xlsx')
    initial_structures = list(initial_structures['SMILES'])
    #load generated structures
    generated_structures = pd.read_excel('../Data/Proposed_structures_with_AI_colchicyne_tanimoto_similarity_.xlsx')
    generated_structures = list(generated_structures['AI_generated_SMILES'])

    #SYBA application
    #SYBA classifier compilation
    mols_ini = [Chem.MolFromSmiles(smi) for smi in initial_structures]
    mols_gen = [Chem.MolFromSmiles(smi) for smi in generated_structures]
    syba = SybaClassifier()
    syba.fitDefaultScore()
    SYBA_score_to_initial_structures = [syba.predict(mol=mol) for mol in mols_ini]
    SYBA_score_to_generated_structures = [syba.predict(mol=mol) for mol in mols_gen]

    threshold = min(SYBA_score_to_initial_structures)
    print('The minimal SYBA score is: '+str(threshold))
    df_gen = pd.DataFrame(data=generated_structures, columns=['SMILES'])
    df_gen['SYBA score'] = SYBA_score_to_generated_structures

    df_gen_fin = df_gen[df_gen['SYBA score'] > threshold]
    df_gen_fin = df_gen_fin.round({'SYBA score': 2})

    #calculate molecular descriptors
    mol_objs = [Chem.MolFromSmiles(smi) for smi in df_gen_fin['SMILES']] 

    calculate_descriptors = True
    if calculate_descriptors == True:
        calc = Calculator(descriptors, ignore_3D=True)
        molecular_descriptors = calc.pandas(mol_objs)
        molecular_descriptors = molecular_descriptors.applymap(is_morder_missing)
        molecular_descriptors['SMILES'] = df_gen_fin['SMILES']
        molecular_descriptors.to_excel('../Data/Final_selection.xlsx')
    else:
        molecular_descriptors = pd.read_excel('../Data/Final_selection.xlsx')

    #assign predicted value
    df_gen_fin['A549 [nM]'] = 0
    df_gen_fin['BALB/3T3 [nM]'] = 0
    df_gen_fin['LoVo [nM]'] = 0
    df_gen_fin['LoVo/DX [nM]'] = 0
    df_gen_fin['MCF-7 [nM]'] = 0

    try:
        #A549
        model = joblib.load('../Activity/Random_forest/random_forest_model_17_estimators_A549.joblib')
        mob = model.predict(molecular_descriptors[list(model.feature_names_in_)])
        converted = pred_model.inverse_transform(mob)
        df_gen_fin['A549 [nM]'] = converted
        
        #BALB/3T3
        model = joblib.load('../Activity/Random_forest/random_forest_model_19_estimators_BALB_3T3.joblib')
        mob = model.predict(molecular_descriptors[list(model.feature_names_in_)])
        converted = pred_model.inverse_transform(mob)
        df_gen_fin['BALB/3T3 [nM]'] = converted

        #LoVo
        model = joblib.load('../Activity/Random_forest/random_forest_model_18_estimators_LoVo.joblib')
        mob = model.predict(molecular_descriptors[list(model.feature_names_in_)])
        converted = pred_model.inverse_transform(mob)
        df_gen_fin['LoVo [nM]'] = converted

        #LoVo/DX
        model = joblib.load('../Activity/Random_forest/random_forest_model_14_estimators_LoVo_DX.joblib')
        mob = model.predict(molecular_descriptors[list(model.feature_names_in_)])
        converted = pred_model.inverse_transform(mob)
        df_gen_fin['LoVo/DX [nM]'] = converted

        #MCF-7
        model = joblib.load('../Activity/Random_forest/random_forest_model_3_estimators_MCF-7.joblib')
        mob = model.predict(molecular_descriptors[list(model.feature_names_in_)])
        converted = pred_model.inverse_transform(mob)
        df_gen_fin['MCF-7 [nM]'] = converted

    except:
        print("Error with predicted value...")


    try:

        df_gen_fin = df_gen_fin.drop_duplicates(subset=['SYBA score', 'A549 [nM]', 'BALB/3T3 [nM]', 'LoVo [nM]', 'LoVo/DX [nM]', 'MCF-7 [nM]'], keep='first')
        
        df_gen_fin['Mol Image'] = [Chem.MolFromSmiles(smi) for smi in df_gen_fin['SMILES']]

        PandasTools.SaveXlsxFromFrame(df_gen_fin, '../Data/Whole_report.xlsx', molCol='Mol Image')

    except:
        print("Error when inserting images...")