# Colchicine_ML
 Machine learning (ML) for colchicine derivatives, anticancer activity

## Order of usage
    1. Generative neural network creation
        1.1. Training data preparation: ../New_structures/training_data_preparation.py
        1.2. Neural network creation: ../New_structures/neural_network.py
        1.3. Neural network loss plot: ../Neural_network/Neural_network.ipynb
    2. Prediction and analysis of new structures
        2.1. Prediction of new structures: ../New_structures/do_predictions.py
        2.2 Colchcine core selection: ../New_structures/Preserve_colchicine_similar_structures.ipynb
        2.3. PubChem search for generated structures: ../New_structures/PubChemPy_search.ipynb
        2.4. SYBA selection: ../New_structures/SYBA_selection.py
        2.5. PubChem search after SYBA selection: ../New_structures/PubChemPy_search-selected_structures.ipynb
        2.6. Chemical space analysis based on molecular fingerprints: ../New_structures/t-SNE.ipynb
        2.7. Stereochemical selection: ../New_structures/Sterochemistry_selection.ipynb
    3. Creation of predictive models based on molecular descriptors and prediction of the feature of interest, $\IC_50$ of each of the cell lines analyzed, namely A549, LoVo, LoVo/DX, BALB/3T3 and MCF-7; 5 ML methodologies are explored - Multiple Linear Regression (MLR), Support Vector Machine (SVM), KNeighbors regression (KNN), Decision Tree regression (DT), Random forest regression (RF)
        3.1. $\IC_50$ preparation: ../Activity/Data_transformation.ipynb
        3.2. Correlation thresholds (molecular descriptors to targets): ../Activity/Correlation_to_targets.ipynb
        3.3. A549: ../Activity/Models_A549_*.ipynb
        3.4. LoVo: ../Activity/Models_LoVo_*.ipynb
        3.5. LoVo/DX: ../Activity/Models_LoVo_DX_*.ipynb
        3.6. BALB/3T3: ../Activity/Models_BALB_3T3_*.ipynb
        3.7. MCF-7: ../Activity/Models_MCF-7_*.ipynb
        3.8. Pick up the best models for each of the targets: ../Activity/Load data and analyze results.ipynb
        3.9. Predict $\IC_50$ for the newly created structures: ../Activity/Activity_prediction.ipynb

## The results storage
    The results are stored in the `Data` folder.


## The used libraries are (requirements, 20.10.2023):
    conda create --name cheminf_gpu
    conda install tensorflow-gpu==2.6.0
    pip install rdkit==2022.9.3
    pip install selfies==2.1.1
    pip install xlsxwriter==3.0.3
    pip install pubchempy==1.0.4
    pip install pandas
    pip install openpyxl==3.0.10
    pip install jupyter notebook
    pip install pyarrow
    conda install fastparquet
    pip install scikit-learn==1.2.0
    pip install keras==2.6.*
    pip install hyperopt==0.2.7
    pip install mordred==1.2.0
    pip install xgboost==1.7.2
    pip install seaborn==0.12.2
    SYBA library is installed by downloading the https://github.com/lich-uct/syba, running "cd syba" and prompting "python setup.py install"
