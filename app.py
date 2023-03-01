import deepchem as dc
import streamlit as st
import numpy as np
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import pandas as pd
import pyperclip

# Models for CYP-450 inhibitors
model_1A2_inh = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_1A2_inh')
model_2C9_inh = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_2C9_inh')
model_2C19_inh = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_2C19_inh')
model_2D6_inh = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_2D6_inh')
model_3A4_inh = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_3A4_inh')

# Models for CYP-450 substrates
model_1A2_sub = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_1A2_sub')
model_2C9_sub = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_2C9_sub')
model_2C19_sub = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_2C19_sub')
model_2D6_sub = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_2D6_sub')
model_3A4_sub = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='models/model_3A4_sub')


model_1A2_inh.restore() 
model_2C9_inh.restore()  
model_2C19_inh.restore() 
model_2D6_inh.restore()  
model_3A4_inh.restore()  

model_1A2_sub.restore()  
model_2C9_sub.restore()  
model_2C19_sub.restore() 
model_2D6_sub.restore()  
model_3A4_sub.restore()  

# Make a list of all models


models = [(model_1A2_inh, 'CYP1A2 Inhibitor'), 
          (model_2C9_inh, 'CYP2C9 Inhibitor'),
          (model_2C19_inh, 'CYP2C19 Inhibitor'),
          (model_2D6_inh, 'CYP2D6 Inhibitor'),
          (model_3A4_inh, 'CYP3A4 Inhibitor'),
          (model_1A2_sub, 'CYP1A2 Substrate'),
          (model_2C9_sub, 'CYP2C9 Substrate'),
          (model_2C19_sub, 'CYP2C19 Substrate'),
          (model_2D6_sub, 'CYP2D6 Substrate'),
          (model_3A4_sub, 'CYP3A4 Substrate')]



st.write("""
# Cytochrome-P450 metabolism prediction
This app predicts whether a given molecule is an inhibitor and substrate of the five most common cytochrome-P450 enzymes. If two molecules are given,
it would provide predictions for both in side-by-side columns, making it easier to track the possibility of drug interaction between the two. 
""")
st.write('---')

if st.button("Example 1"):
    user_input1 = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"
    user_input2 = "CC(=O)Nc1ccc(O)cc1"
elif st.button("Example 2"):
    user_input1 = "C1=CC(=CC=C1C(=O)NC2=CC=C(C=C2)Cl)Cl"
    user_input2 = "Brc1cc(cc(Br)c1O)C(=O)c2c3ccccc3oc2CC"
else:
    user_input1 = st.text_input("Please enter molecule 1 in SMILES format", key="drug_1")
    user_input2 = st.text_input("Please enter molecule 2 in SMILES format", key="drug_2")

col1, col2 = st.columns(2)

with col1:
    st.write('---')
    st.write("User input 1: ", user_input1, key="drug_1")
    if user_input1:
        mol = Chem.MolFromSmiles(user_input1)
        if mol is not None:
            img = Draw.MolToImage(mol)
            with BytesIO() as output:
                img.save(output, format="PNG")
                image_bytes = output.getvalue()
            st.image(Image.open(BytesIO(image_bytes)))
        else:
            st.write("Invalid SMILES string")
    st.write('---')

with col2:
    st.write('---')
    st.write("User input 2: ", user_input2, key="drug_2")
    if user_input2:
        mol = Chem.MolFromSmiles(user_input2)
        if mol is not None:
            img = Draw.MolToImage(mol)
            with BytesIO() as output:
                img.save(output, format="PNG")
                image_bytes = output.getvalue()
            st.image(Image.open(BytesIO(image_bytes)))
        else:
            st.write("Invalid SMILES string")
    st.write('---')


# apply the ConvMolFeaturizer on the user_input using DeepChem and apply the model
smile_1 = user_input1
smile_2 = user_input2

# create an empty DataFrame to store the results
results_df_1 = pd.DataFrame(columns=['Model name', 'Prediction'])
results_df_2 = pd.DataFrame(columns=['Model name', 'Prediction'])

def predict(smile, model):
    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer(smile)
    pred = model.predict_on_batch(X)
    return np.argmax(pred)

with col1:
    st.write("Predictions for Molecule 1 [0 = inactive, 1 = active]")
    if user_input1:
        for model, model_name in models: # loop over each model and make a prediction
            prediction = predict(smile_1, model)
            results_df_1 = results_df_1.append({'Model name': model_name, 'Prediction': prediction}, ignore_index=True)
        st.write(results_df_1.to_html(index=False), unsafe_allow_html=True)

with col2:
    st.write("Predictions for Molecule 2 [0 = inactive, 1 = active]")
    if user_input2:
        for model, model_name in models: # loop over each model and make a prediction
            prediction = predict(smile_2, model)
            results_df_2 = results_df_2.append({'Model name': model_name, 'Prediction': prediction}, ignore_index=True)
        st.write(results_df_2.to_html(index=False), unsafe_allow_html=True)


