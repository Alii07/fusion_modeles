import streamlit as st
import camelot
import pandas as pd
import base64
from io import StringIO

# Fonction pour extraire les tableaux du PDF
@st.cache_data
def process_pdf(pdf_bytes):
    # Enregistre le PDF temporairement et extrait les tableaux
    with open("tmp.pdf", "wb") as file:
        file.write(pdf_bytes.read())
    tables = camelot.read_pdf("tmp.pdf", pages="all")
    return tables

# Fonction pour permettre le téléchargement des tableaux sous format CSV
def display_download_options(df):
    """Télécharge un DataFrame en CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Télécharger le fichier CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# Application principale
def main():
    st.title("Extraction de tableaux à partir de PDF avec Camelot")
    
    uploaded_pdf = st.file_uploader("Téléverser un fichier PDF", type="pdf")
    
    if uploaded_pdf is not None:
        tables = process_pdf(uploaded_pdf)
        
        st.write(f"{len(tables)} tableaux trouvés.")
        
        # Parcourir chaque tableau extrait
        for i, table in enumerate(tables):
            df = table.df  # Convertir le tableau en DataFrame
            st.write(f"Tableau {i+1}")
            st.dataframe(df)  # Afficher le tableau
            
            # Option de téléchargement pour chaque tableau
            display_download_options(df)
    
    else:
        st.write("Veuillez téléverser un fichier PDF contenant des tableaux.")

if __name__ == "__main__":
    main()