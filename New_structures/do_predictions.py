# General Imports
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from tensorflow import keras
import sys
from pathlib import Path
SELFIES_coder_path = Path("../SELFIES_coder")
sys.path.append(SELFIES_coder_path.as_posix())
import SELFIES_coder as SELFIES_CODER
import selfies as sf
import shutil

if __name__ == "__main__":

    ##load data
    filename = 'Kolchicyna_prepared_data'

    df = pd.read_excel('../Data/'+str(filename)+'.xlsx')
    df_list = list(df['SMILES'])

    mol_seq_lat = []
    lat_states = []
    predictor = []
    
    mol_seq_lat.append("../New_structures/mol_seq2lat.h5")
    
    lat_states.append("../New_structures/lat2state.h5")
    
    predictor.append("../New_structures/samplemodel.h5")


    embed = 99 #from the very first model second dimension +1 !!!

    import json
    f = open('../New_structures/mol_seq_to_int.json')
    char_to_int = json.load(f)
    charset = char_to_int.keys()
    to_be_used = len(charset)

    import json
    f = open('../New_structures/int_to_mol_seq.json')
    int_to_char = json.load(f)
    int_to_char = {int(key):int_to_char[key] for key in int_to_char} #removed '' -string form of data

    import json
    f = open('../New_structures/mol_seq_to_SELFIES.json')
    One_hot_to_SELFIES = json.load(f)

    import json
    f = open('../New_structures/SELFIES_to_mol_seq.json')
    SELFIES_to_one_hot = json.load(f)


    data = []
    for SMI in list(df_list):
        new = SELFIES_CODER.SMILES_to_SELFIES(SMI)
        data.append(new)
    data_ = SELFIES_CODER.make_array(data)

    translation = SELFIES_CODER.translate_SELFIES_array_into_one_hot(data_, SELFIES_to_one_hot)

    decoded_SELFIES = []
    for i in range(len(translation)):
        decoded_ = SELFIES_CODER.convert_back_to_SEFLIES(translation[i], One_hot_to_SELFIES)
        decoded_SELFIES.append(decoded_)
    print(decoded_SELFIES[0])

    shap = len(list(df_list))

    #vectorization of SMILES code
    def vectorize(mol_seqs, shap):
            one_hot =  np.zeros((shap, embed , len(charset)),dtype=np.int8)
            for i,mol_seq in enumerate(mol_seqs):
                #encode the startchar
                one_hot[i,0,char_to_int["!"]] = 1
                #encode the rest of the chars
                for j,c in enumerate(mol_seq):
                    one_hot[i,j+1,char_to_int[c]] = 1
                #Encode endchar
                one_hot[i,len(mol_seq)+1:,char_to_int["E"]] = 1
            #Return two, one for input and the other for output
            return one_hot[:,0:-1,:], one_hot[:,1:,:]
    mol_seqs_pred_ = translation

    X_train = vectorize(mol_seqs_pred_, shap)


    #going back from vectorized form to redable string
    string_test = "".join([int_to_char[idx] for idx in np.argmax(X_train[0][1,:,:], axis=1)])
    print(string_test)

    decoded = SELFIES_CODER.convert_back_to_SEFLIES(string_test, One_hot_to_SELFIES)
    print(decoded)

    decoded_two = SELFIES_CODER.convert_back_to_SEFLIES(mol_seqs_pred_[1], One_hot_to_SELFIES)
    print(decoded_two)

    print("Correct encoding-decoding: "+str(decoded == decoded_two))

    new_smi_approach = vectorize(mol_seqs_pred_, shap)

    mol_seq_to_latent_model = keras.models.load_model(mol_seq_lat[0])

    latent_to_states_model = keras.models.load_model(lat_states[0])

    sample_model = keras.models.load_model(predictor[0])

    x_latent_new = mol_seq_to_latent_model.predict(new_smi_approach[0:1])


    def latent_to_mol_seq(latent):
        #decode states and set Reset the LSTM cells with them
        states = latent_to_states_model.predict(latent)
        sample_model.layers[1].reset_states(states=[states[0],states[1]])
        #Prepare the input char
        startidx = char_to_int["!"]
        samplevec = np.zeros((1,1,len(charset))) #last value should be eddited
        samplevec[0,0,startidx] = 1
        mol_seq = ""
        #Loop and predict next char
        for i in range(embed): #range is the smiles length - should be properly fixed
            o = sample_model.predict(samplevec)
            sampleidx = np.argmax(o)
            samplechar = int_to_char[sampleidx]
            if samplechar != "E":
                mol_seq = mol_seq + int_to_char[sampleidx]
                samplevec = np.zeros((1,1,len(charset))) #last value should be eddited
                samplevec[0,0,sampleidx] = 1
            else:
                break
        return mol_seq
    
    molecular_sequence = latent_to_mol_seq(x_latent_new[0:1])

    def SELFIES_to_SMILES(SELFIES):
    
        smi = sf.decoder(SELFIES)
        
        return smi
    
    #Sample around the latent wector
    scale = 0.20 #0.10
    mols = []

    for i in range(x_latent_new.shape[0]):
        for elem in range(20):
            latent_r = x_latent_new[i:i+1] + scale*(np.random.randn(x_latent_new[i:i+1].shape[1]))
            mol_seq = latent_to_mol_seq(latent_r)
            decoded = SELFIES_CODER.convert_back_to_SEFLIES(mol_seq, One_hot_to_SELFIES)
            new_SMI = SELFIES_to_SMILES(decoded)
            mol = Chem.MolFromSmiles(new_SMI)
            if mol:
                mols.append(new_SMI)
            
                print(i+1, "run ",elem+1, 'try, ', 'Correct molecule...', ' ', new_SMI)
            else:
                print(i+1, "run ",elem+1, 'try,', "ERROR", ' ', new_SMI)

    print('All AI-generated structures: '+str(len(mols)))


    def remove_duplicates_from_list(x_list):
    
        return list(dict.fromkeys(x_list))
    
    mols = remove_duplicates_from_list(mols)
    print('Unique structures: '+str(len(mols)))

    dataframe_created_molecules = pd.DataFrame(mols, columns=["AI_generated_SMILES"])
    dataframe_created_molecules.to_excel('../Data/AI_generated_Molecules_0_2_.xlsx')