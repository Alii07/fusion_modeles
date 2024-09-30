import concurrent.futures
import shutil
import tempfile
import csv
from io import StringIO
import os
import pandas as pd
from PyPDF2 import PdfReader
import streamlit as st
import camelot

def extract_table_from_pdf(pdf_file_path, edge_tol, row_tol, pages):
    try:
        tables_stream = camelot.read_pdf(
            pdf_file_path,
            flavor='stream',
            pages=pages,
            strip_text='\n',
            edge_tol=edge_tol,
            row_tol=row_tol
        )
        return tables_stream

    except Exception as e:
        return None

@st.cache_data
def save_table_to_memory_csv(df):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

def check_second_line(file_content, required_elements):
    file_like_object = StringIO(file_content)
    reader = csv.reader(file_like_object)
    next(reader)  # Ignorer la première ligne (header)
    second_line = next(reader, None)  # Lire la deuxième ligne
    if second_line and all(elem in second_line for elem in required_elements):
        return True
    return False

def split_columns(header, second_line, required_elements):
    required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
    other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
    return required_indices, other_indices

# Fonction pour traiter les pages du PDF
def process_pages(pdf_file_path, edge_tol, row_tol, page):
    tables_stream = extract_table_from_pdf(pdf_file_path, edge_tol, row_tol, pages=page)
    results = []
    if tables_stream is not None and len(tables_stream) > 0:
        largest_table = max(tables_stream, key=lambda t: t.df.shape[0] * t.df.shape[1])
        df_stream = largest_table.df
        df_stream.replace('\n', '', regex=True, inplace=True)
        df_stream.fillna('', inplace=True)
        page_number = largest_table.parsing_report['page']

        if 'Montant Sal.Taux' in df_stream.iloc[0].values:
            refined_tables = extract_table_from_pdf(pdf_file_path, edge_tol=500, row_tol=5, pages=str(page_number))
            if refined_tables is not None and len(refined_tables) > 0:
                largest_table = max(refined_tables, key=lambda t: t.df.shape[0] * t.df.shape[1])
                df_stream = largest_table.df
                df_stream.replace('\n', '', regex=True, inplace=True)
                df_stream.fillna('', inplace=True)
        
        results.append((page_number, df_stream))
    
    return results

# Titre de l'application Streamlit
st.title("Extraction de bulletins de paie à partir de PDF")
uploaded_pdf = st.file_uploader("Téléverser un fichier PDF", type=["pdf"])
uploaded_file_1 = st.file_uploader("1er fichier excel", type=['xlsx', 'xls'])
uploaded_file_2 = st.file_uploader("2nd fichier excel", type=['xlsx', 'xls'])

if uploaded_pdf is not None and uploaded_file_1 is not None and uploaded_file_2 is not None:
    # Créer un fichier temporaire pour le PDF téléchargé
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_pdf.read())  
        temp_pdf_path = temp_pdf.name
    
    # Dictionnaire pour stocker les fichiers CSV en mémoire
    csv_files = {}

    reader = PdfReader(temp_pdf_path)
    total_pages = len(reader.pages)

    current_page_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    max_workers = 4  # Limite du nombre de threads/process

    st.write(f"Extraction des tableaux pour toutes les {total_pages} pages...")

    # Utilisation de `ProcessPoolExecutor` pour traiter les pages en parallèle
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        other_page_futures = {executor.submit(process_pages, temp_pdf_path, 300, 3, str(page)): page for page in range(1, total_pages + 1)}
        
        for future in concurrent.futures.as_completed(other_page_futures):
            page = other_page_futures[future]
            try:
                results = future.result()
                for page_number, df_stream in results:
                    csv_content = save_table_to_memory_csv(df_stream)
                    csv_files[f"table_page_{page_number}.csv"] = csv_content
                
                current_page_count += 1
                progress_value = current_page_count / total_pages
                progress_bar.progress(progress_value)
                status_text.text(f"Traitement : {min(current_page_count, total_pages)}/{total_pages} pages traitées")
                
            except Exception as e:
                st.write(f"Erreur lors du traitement des pages {page}: {e}")

    st.write("Extraction des tableaux terminée.")

    # Liste des éléments requis
    required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']
    required_elements2 = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']

    filtered_files = []

    # Parcourir les fichiers CSV en mémoire et filtrer selon `check_second_line`
    for filename, csv_content in csv_files.items():
        if check_second_line(csv_content, required_elements) or check_second_line(csv_content, required_elements2):
            filtered_files.append((filename, csv_content))  # Ajouter le nom et le contenu du fichier filtré
    
    st.write("Les fichiers CSV filtrés sont prêts à être utilisés.")

    # Exemple d'affichage des fichiers filtrés
    for filename, csv_content in filtered_files:
        st.write(f"Fichier filtré : {filename}")
        st.download_button(f"Télécharger {filename}", csv_content, filename)
