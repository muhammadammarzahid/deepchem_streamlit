import deepchem as dc
import streamlit as st
import numpy as np
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

model = dc.models.GraphConvModel(n_tasks=1, mode='classification', model_dir='model_dir')

st.write("""
# Cytochrome-P450 metabolism prediction

This app predicts the metabolism of drugs.
""")
st.write('---')

# create a text input box and store the user's input in a variable
user_input = st.text_input("Please enter the molecule in SMILES format")

# display the user's input on the screen
st.write("User input:", user_input)

# convert user_input to a RDKit molecule
if user_input:
    mol = Chem.MolFromSmiles(user_input)
    if mol is not None:
        # generate an image of the molecule
        img = Draw.MolToImage(mol)
        # convert the image to bytes
        with BytesIO() as output:
            img.save(output, format="PNG")
            image_bytes = output.getvalue()
        # display the image on the screen
        st.image(Image.open(BytesIO(image_bytes)))
    else:
        st.write("Invalid SMILES string")

# apply the ConvMolFeaturizer on the user_input using DeepChem and apply the model
def predict(smile, model):
    featurizer = dc.feat.ConvMolFeaturizer()
    X = featurizer(smile)
    pred = model.predict_on_batch(X)
    return np.argmax(pred)

if user_input:
    smile = user_input
    prediction = predict(smile, model)

    if  prediction == 1:
        st.write("The provided molecule is an inhibitor of CYP3A4")
    else:
        st.write("The provided molecule is NOT an inhibitor of CYP3A4")
