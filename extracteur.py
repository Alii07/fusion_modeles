import camelot
import pandas as pd
import os
import re
import csv
import shutil
from PyPDF2 import PdfReader
import streamlit as st
import random
import tempfile
import glob
import streamlit as st
import camelot
import pandas as pd
import os
import tempfile
import concurrent.futures

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
        #st.write(f"Erreur lors de l'extraction des tableaux à partir du PDF pour les pages {pages}: {str(e)}")
        return None

def save_table_to_csv(df, file_path):
    df.to_csv(file_path, index=False, encoding='utf-8')

required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']
required_elements2 = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']

code_libelle_dict = {}
errors = []
rubriques_codes_from_files = set()
main_table_path = './merged_output.csv'
cumul_path = './Cumul de janvier à juin.xlsx'
info_salaries_path = './Information sur les salariés.xlsx'
absences_path = './Absences.csv'
final_output_csv_path = './final_output.csv'

filtered_files = []
csv_directory = "./CSV3"

output_directory = os.path.join(csv_directory, "bulletins")
os.makedirs(output_directory, exist_ok=True)

clean_output_directory = os.path.join(output_directory, "bulletins_propres")
os.makedirs(clean_output_directory, exist_ok=True)

rest_output_directory = os.path.join(output_directory, "restes_tableaux")
os.makedirs(rest_output_directory, exist_ok=True)

cleaner_output_directory = os.path.join(clean_output_directory, "bulletins_propres_structurés")
os.makedirs(cleaner_output_directory, exist_ok=True)

output_directory = os.path.join(csv_directory, "bulletins")
os.makedirs(output_directory, exist_ok=True)

output_directory_mat = os.path.join(csv_directory, "matricules")
os.makedirs(output_directory_mat, exist_ok=True)

processed_directory = './CSV3/bulletins/bulletins_propres/bulletins_propres_structurés/processed'
os.makedirs(processed_directory, exist_ok=True)





def transform_and_combine_tables(extracted_tables):

    transformed_data = []

    columns = [
        'CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux Montant Pat.',
        'Du', 'Date', 'Equipe', 'Hor.', 'Abs.'
    ]

    for df in extracted_tables:
        for i in range(len(df)):
            row = {}
            row['CodeLibellé'] = df.iloc[i, 0]
            for j, col_name in enumerate(columns[1:], 1):
                if j < len(df.columns):
                    row[col_name] = df.iloc[i, j]
                else:
                    row[col_name] = ''
            transformed_data.append(row)

    combined_df = pd.DataFrame(transformed_data, columns=columns)
    return combined_df


def clean_text(text):
    cleaned_text = text.replace('NaN', '')

    cleaned_text = cleaned_text.replace('\n', ' ')

    return cleaned_text
# Fonction pour gérer le traitement d'une seule page (ou d'un groupe de pages)
def process_pages(pdf_file_path, edge_tol, row_tol, page):
    # Extraction initiale des tableaux
    tables_stream = extract_table_from_pdf(pdf_file_path, edge_tol, row_tol, pages=page)

    results = []

    if tables_stream is not None and len(tables_stream) > 0:
        # Sélectionner le plus grand tableau (en nombre de cellules : lignes * colonnes)
        largest_table = max(tables_stream, key=lambda t: t.df.shape[0] * t.df.shape[1])

        # Extraire le DataFrame du plus grand tableau
        df_stream = largest_table.df
        df_stream.replace('\n', '', regex=True, inplace=True)
        df_stream.fillna('', inplace=True)
        page_number = largest_table.parsing_report['page']

        # Vérifier si 'Montant Sal.Taux' est dans les colonnes
        if 'Montant Sal.Taux' in df_stream.iloc[0].values:
            print(f"Colonne 'Montant Sal.Taux' détectée sur la page {page_number}. Ré-extraction avec les nouveaux paramètres.")

            # Ré-extraire avec les nouveaux paramètres pour cette page
            refined_tables = extract_table_from_pdf(pdf_file_path, edge_tol=500, row_tol=5, pages=str(page_number))

            # Si une ré-extraction réussie est effectuée, sélectionner à nouveau le plus grand tableau
            if refined_tables is not None and len(refined_tables) > 0:
                largest_table = max(refined_tables, key=lambda t: t.df.shape[0] * t.df.shape[1])
                df_stream = largest_table.df
                df_stream.replace('\n', '', regex=True, inplace=True)
                df_stream.fillna('', inplace=True)

        # Ajouter le tableau sélectionné et son numéro de page aux résultats
        results.append((page_number, df_stream))

    return results


# Interface de Streamlit pour le téléversement de fichiers
st.title("Extraction de bulletins de paie à partir de PDF")
uploaded_pdf = st.file_uploader("Téléverser un fichier PDF", type=["pdf"])
uploaded_file_1 = st.file_uploader("1er fichier excel", type=['xlsx', 'xls'])
uploaded_file_2 = st.file_uploader("2nd fichier excel", type=['xlsx', 'xls'])

if uploaded_pdf is not None and uploaded_file_1 is not None and uploaded_file_2 is not None:
    # Crée un fichier temporaire pour enregistrer le PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_pdf.read())  # Écrire le contenu du fichier téléchargé dans le fichier temporaire
        temp_pdf_path = temp_pdf.name  # Obtenir le chemin du fichier temporaire

    # Dossier de sortie pour les fichiers CSV
    output_dir = "CSV3"
    os.makedirs(output_dir, exist_ok=True)

    # Dictionnaire pour stocker les DataFrames pour le téléchargement
    csv_files = {}

    # Lecture du fichier PDF pour obtenir le nombre total de pages
    reader = PdfReader(temp_pdf_path)
    total_pages = len(reader.pages)

    current_page_count = 0

    # Afficher une barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Taille du lot (nombre de pages à traiter en parallèle)
    batch_size = 10  # Nombre de pages à traiter dans chaque lot (à ajuster en fonction de vos ressources)

    # Limiter le nombre de processus simultanés pour éviter de surcharger le CPU
    max_workers = 4  # Nombre maximal de processus simultanés

    st.write(f"Extraction des tableaux pour toutes les {total_pages} pages...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Soumission des tâches pour chaque page
        other_page_futures = {executor.submit(process_pages, temp_pdf_path, 300, 3, str(page)): page for page in range(1, total_pages + 1)}
        
        for future in concurrent.futures.as_completed(other_page_futures):
            page = other_page_futures[future]
            try:
                results = future.result()
                for page_number, df_stream in results:
                    # Enregistrer le DataFrame extrait en CSV
                    stream_output_csv_file = os.path.join(output_dir, f"table_page_{page_number}.csv")
                    save_table_to_csv(df_stream, stream_output_csv_file)
                    csv_files[f"table_page_{page_number}.csv"] = df_stream
                
                # Mettre à jour la barre de progression et l'état
                current_page_count += 1
                progress_value = current_page_count / total_pages
                if progress_value > 1.0:
                    progress_value = 1.0  # S'assurer que la progression ne dépasse pas 1.0
                progress_bar.progress(progress_value)
                status_text.text(f"Traitement : {min(current_page_count, total_pages)}/{total_pages} pages traitées")
                
            except Exception as e:
                st.write(f"Erreur lors du traitement des pages {page}: {e}")

    st.write("Extraction des tableaux terminée.")
    # Ajouter des boutons de téléchargement pour chaque fichier CSV
    #st.write("Télécharger les fichiers CSV extraits :")


    def check_second_line(file_path, required_elements):
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Ignore la première ligne
            second_line = next(reader, None)  # Lire la deuxième ligne
            if second_line and all(elem in second_line for elem in required_elements):
                return True
        return False
    
    def split_columns(header, second_line, required_elements):
        required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
        other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
        return required_indices, other_indices



    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_directory, filename)
            if check_second_line(file_path, required_elements) or check_second_line(file_path, required_elements2):
                filtered_files.append(file_path)

    # Copier les fichiers CSV filtrés dans le sous-répertoire
    for file in filtered_files:
        shutil.copy(file, output_directory)

    st.write("Fichiers CSV filtrés enregistrés dans CSV3/bulletins.")

    def check_for_mat(text):
        return 'Mat:' in text

    def extract_matricules(text):
        matricules = set()
        for line in text.split('\n'):
            if 'Mat:' in line:
                # Extraire la partie après "Mat:" et avant "/ Gest:" ou le premier espace
                start = line.find('Mat:') + len('Mat:')
                end = line.find('/ Gest:', start)
                if end == -1:
                    end = len(line)
                matricule = line[start:end].strip()
                matricules.add(matricule)
        return matricules

    reader = PdfReader(uploaded_pdf)
    all_matricules = set()

    for page in reader.pages:
        text = page.extract_text()
        if check_for_mat(text):
            all_matricules.update(extract_matricules(text))


    matricules_file_path = os.path.join(output_directory_mat, "matricules.csv")
    with open(matricules_file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Matricule"])
        writer.writerow([])  # Ajouter une ligne vide après l'en-tête
        for matricule in sorted(all_matricules):
            writer.writerow([matricule])

    st.write(f"Matricules distinctes enregistrées dans {matricules_file_path}.")

    def rename_second_taux(file_path):
        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            lines = list(reader)

        if len(lines) > 1:
            second_line = lines[1]
            taux_indices = [i for i, col in enumerate(second_line) if col == 'Taux']
            if len(taux_indices) > 1:
                second_line[taux_indices[1]] = 'Taux 2'

        with open(file_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(lines)


    def check_second_line(file_path, required_elements):
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Ignore la première ligne
            second_line = next(reader, None)  # Lire la deuxième ligne
            if second_line and all(elem in second_line for elem in required_elements):
                return True
        return False

    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_directory, filename)
            if check_second_line(file_path, required_elements) or check_second_line(file_path, required_elements2):
                rename_second_taux(file_path)
                filtered_files.append(file_path)

    for file in filtered_files:
        with open(file, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Lire la première ligne (en-tête)
            second_line = next(reader)  # Lire la deuxième ligne (colonnes requises)
            required_indices, other_indices = split_columns(header, second_line, required_elements + required_elements2)

            # Créer les chemins de sortie pour les fichiers divisés
            clean_file_path = os.path.join(clean_output_directory, os.path.basename(file))
            rest_file_path = os.path.join(rest_output_directory, os.path.basename(file))

            # Écrire les fichiers avec les colonnes requises
            with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                writer = csv.writer(clean_outfile)
                writer.writerow([header[i] for i in required_indices])
                writer.writerow([second_line[i] for i in required_indices])
                for row in reader:
                    writer.writerow([row[i] for i in required_indices])

            # Rewind the reader to read the file again for the rest columns
            infile.seek(0)
            reader = csv.reader(infile)
            header = next(reader)  # Lire la première ligne (en-tête)
            second_line = next(reader)  # Lire la deuxième ligne (colonnes requises)

            # Écrire les fichiers avec les autres colonnes
            with open(rest_file_path, mode='w', newline='', encoding='utf-8') as rest_outfile:
                writer = csv.writer(rest_outfile)
                writer.writerow([header[i] for i in other_indices])
                writer.writerow([second_line[i] for i in other_indices])
                for row in reader:
                    writer.writerow([row[i] for i in other_indices])

    st.write("Fichiers CSV divisés et sauvegardés dans les répertoires appropriés.")

    # Fonction pour diviser les colonnes en deux groupes
    def split_columns(header, second_line, required_elements):
        required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
        other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
        return required_indices, other_indices

    # Fonction pour renommer la deuxième occurrence de "Taux" en "Taux 2"
    def rename_second_taux(file_path):
        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            lines = list(reader)

        if len(lines) > 1:
            second_line = lines[1]
            taux_indices = [i for i, col in enumerate(second_line) if col == 'Taux']
            if len(taux_indices) > 1:
                second_line[taux_indices[1]] = 'Taux 2'

        with open(file_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(lines)




    def transform_to_two_lines(file_path, required_elements, required_elements2):
        headers_row = []
        values_row = []
        code_libelle_dict = {}
        errors = []

        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Lire la première ligne (en-tête)
            second_line = next(reader)  # Lire la deuxième ligne (colonnes requises)

            # Vérifier la présence des colonnes 'Code' et 'Libellé'
            code_index = second_line.index('Code') if 'Code' in second_line else None
            libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
            codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

            rubriques = []
            if codelibelle_index is not None:
                for row in reader:
                    rubrique = row[codelibelle_index]
                    if rubrique and rubrique[0].isdigit():
                        code = rubrique[:4]  # Tronquer à 4 caractères si commence par un chiffre
                        libelle = rubrique[5:].strip() if len(rubrique) > 4 else ''
                        if libelle:  # Ajouter seulement si le libellé n'est pas vide
                            code_libelle_dict[code] = libelle
                        rubriques.append(rubrique)
                        headers_row.extend([
                            f"{rubrique}Base",
                            f"{rubrique}Taux",
                            f"{rubrique}Montant Sal.",
                            f"{rubrique}Taux 2",
                            f"{rubrique}Montant Pat."
                        ])
            elif code_index is not None and libelle_index is not None:
                for row in reader:
                    rubrique = f"{row[code_index]}{row[libelle_index]}"
                    if rubrique and rubrique[0].isdigit():
                        code = rubrique[:4]  # Tronquer à 4 caractères si commence par un chiffre
                        libelle = row[libelle_index].strip()
                        if libelle:  # Ajouter seulement si le libellé n'est pas vide
                            code_libelle_dict[code] = libelle
                        rubriques.append(rubrique)
                        headers_row.extend([
                            f"{rubrique}Base",
                            f"{rubrique}Taux",
                            f"{rubrique}Montant Sal.",
                            f"{rubrique}Taux 2",
                            f"{rubrique}Montant Pat."
                        ])

            # Rewind the reader to read the file again for values
            infile.seek(0)
            next(reader)  # Ignore the first line (header)
            next(reader)  # Ignore the second line (column names)

            # Initialiser la ligne de valeurs avec des cellules vides
            values_row = ['' for _ in range(len(headers_row))]

            # Remplir la ligne des valeurs avec les données correspondantes
            for row in reader:
                if codelibelle_index is not None:
                    code_libelle = row[codelibelle_index]
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  # Tronquer à 4 caractères si commence par un chiffre
                elif code_index is not None and libelle_index is not None:
                    code_libelle = f"{row[code_index]}{row[libelle_index]}"
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  # Tronquer à 4 caractères si commence par un chiffre
                else:
                    errors.append("Les colonnes 'Code' et 'Libellé' ou 'CodeLibellé' sont manquantes")
                    return [headers_row, values_row, code_libelle_dict, errors]  # Sortir avec erreur

                rubrique_base_header = f"{code_libelle}Base"
                try:
                    rubrique_index = headers_row.index(rubrique_base_header) // 5 * 5  # Index de début pour cette rubrique
                    for i, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                        if col in second_line:
                            col_index = second_line.index(col)
                            values_row[rubrique_index + i] = row[col_index]
                except ValueError:
                    # Ajouter une erreur significative et continuer
                    errors.append(f"Erreur dans le fichier {file_path}: En-tête '{rubrique_base_header}' introuvable.")

        return [headers_row, values_row, code_libelle_dict, errors]



    for filename in os.listdir(clean_output_directory):
        if filename.endswith(".csv") and filename != "bulletins_propres":
            file_path = os.path.join(clean_output_directory, filename)

            # Transformer les données en deux lignes
            required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']
            required_elements2 = ['Code', 'Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']

            try:
                headers_row, values_row, new_code_libelle_dict, file_errors = transform_to_two_lines(
                    file_path, required_elements, required_elements2)
                code_libelle_dict.update(new_code_libelle_dict)
                errors.extend(file_errors)

                # Collect all unique rubrique codes from this file
                rubriques_codes_from_files.update(new_code_libelle_dict.keys())

                # Sauvegarder le fichier transformé
                clean_file_path = os.path.join(cleaner_output_directory, "restructured_" + filename)
                with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                    writer = csv.writer(clean_outfile)
                    writer.writerow(headers_row)
                    writer.writerow(values_row)
            except ValueError as e:
                errors.append(f"Erreur dans le fichier {filename}: {e}")

    sorted_code_libelle = sorted(code_libelle_dict.items())

    dictionnaire_file_path = "Dictionnaire2.txt"
    with open(dictionnaire_file_path, mode='w', encoding='utf-8') as dict_file:
        for code, libelle in sorted_code_libelle:
            dict_file.write(f"{code} : {libelle}\n")

    final_codes = set()
    with open(dictionnaire_file_path, mode='r', encoding='utf-8') as dict_file:
        for line in dict_file:
            code = line.split(':')[0].strip()
            final_codes.add(code)

    missing_codes = rubriques_codes_from_files - final_codes

    # Imprimer les codes manquants
    #st.write("Codes de rubriques manquants dans le dictionnaire :")

    # Écrire les erreurs dans un fichier texte
    errors_file_path = "errors.txt"
    with open(errors_file_path, mode='w', encoding='utf-8') as errors_file:
        for error in errors:
            errors_file.write(f"{error}\n")

    #st.write(f"Dictionnaire des codes rubriques et libellés enregistré dans {dictionnaire_file_path}.")
    #st.write(f"Liste des erreurs enregistrée dans {errors_file_path}.")

    # Fonction pour générer un taux selon une certaine logique
    def generer_taux_2(taux_initial):
        return round(random.uniform(taux_initial - 0.2, taux_initial + 0.2), 2)

    # Fonction pour vérifier et convertir une valeur en float, en gérant les valeurs vides
    def safe_float_conversion(value):
        try:
            return float(value)
        except ValueError:
            return None  # Retourne None si la valeur ne peut pas être convertie

    # Fonction pour transformer le tableau en deux lignes et ajouter 'Taux 2'
    def transform_to_two_lines(file_path, required_elements, required_elements2):
        headers_row = []
        values_row = []

        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Lire la première ligne (en-tête)
            second_line = next(reader)  # Lire la deuxième ligne (colonnes requises)

            # Vérifier la présence des colonnes 'Code' et 'Libellé'
            code_index = second_line.index('Code') if 'Code' in second_line else None
            libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
            codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

            rubriques = []
            if codelibelle_index is not None:
                for row in reader:
                    rubrique = row[codelibelle_index]
                    if rubrique not in rubriques:
                        if rubrique and rubrique[0].isdigit():
                            rubrique = rubrique[:4]  # Tronquer à 4 caractères si commence par un chiffre
                        rubriques.append(rubrique)
                        headers_row.extend([
                            f"{rubrique}Base",
                            f"{rubrique}Taux",
                            f"{rubrique}Montant Sal.",
                            f"{rubrique}Taux 2",  # Ajout de la colonne Taux 2
                            f"{rubrique}Montant Pat."
                        ])
            elif code_index is not None and libelle_index is not None:
                for row in reader:
                    rubrique = f"{row[code_index]}{row[libelle_index]}"
                    if rubrique not in rubriques:
                        if rubrique and rubrique[0].isdigit():
                            rubrique = rubrique[:4]  # Tronquer à 4 caractères si commence par un chiffre
                        rubriques.append(rubrique)
                        headers_row.extend([
                            f"{rubrique}Base",
                            f"{rubrique}Taux",
                            f"{rubrique}Montant Sal.",
                            f"{rubrique}Taux 2",  # Ajout de la colonne Taux 2
                            f"{rubrique}Montant Pat."
                        ])

            # Rewind the reader to read the file again for values
            infile.seek(0)
            next(reader)  # Ignore the first line (header)
            next(reader)  # Ignore the second line (column names)

            # Initialiser la ligne de valeurs avec des cellules vides
            values_row = ['' for _ in range(len(headers_row))]

            # Remplir la ligne des valeurs avec les données correspondantes
            for row in reader:
                if codelibelle_index is not None:
                    code_libelle = row[codelibelle_index]
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  # Tronquer à 4 caractères si commence par un chiffre
                elif code_index is not None and libelle_index is not None:
                    code_libelle = f"{row[code_index]}{row[libelle_index]}"
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  # Tronquer à 4 caractères si commence par un chiffre
                else:
                    raise ValueError("Les colonnes 'Code' et 'Libellé' ou 'CodeLibellé' sont manquantes")

                rubrique_index = headers_row.index(f"{code_libelle}Base") // 5 * 5  # Index de début pour cette rubrique
                for i, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                    if col in second_line:
                        col_index = second_line.index(col)
                        values_row[rubrique_index + i] = row[col_index]

                # Générer et ajouter la valeur pour 'Taux 2' si 'Taux' est valide
                taux_value = safe_float_conversion(values_row[rubrique_index + 1])  # 'Taux'
                if taux_value is not None:  # Vérifier que 'Taux' n'est pas vide ou invalide
                    taux_2_value = generer_taux_2(taux_value)  # Générer 'Taux 2'
                    values_row[rubrique_index + 3] = taux_2_value  # Placer 'Taux 2' dans la bonne colonne

        return [headers_row, values_row]

    for filename in os.listdir(clean_output_directory):
        if filename.endswith(".csv") and filename != "bulletins_propres":
            file_path = os.path.join(clean_output_directory, filename)

            # Transformer les données en deux lignes
            required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Montant Pat.']
            required_elements2 = ['Code', 'Libellé', 'Base', 'Taux', 'Montant Sal.', 'Montant Pat.']

            try:
                headers_row, values_row = transform_to_two_lines(file_path, required_elements, required_elements2)

                # Sauvegarder le fichier transformé
                clean_file_path = os.path.join(cleaner_output_directory, "restructured_" + filename)
                with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                    writer = csv.writer(clean_outfile)
                    writer.writerow(headers_row)
                    writer.writerow(values_row)
                    #print(f"Fichier CSV restructuré et sauvegardé: {clean_file_path}")
            except ValueError as e:
                st.write(f"Erreur: {e}")


    st.write("Fichiers CSV restructurés et sauvegardés dans les répertoires appropriés.")


    def convert_to_float(value):
        if value:
            try:
                value = value.replace('.', '').replace(',', '.')
                return float(value)
            except ValueError:
                return None
        return None

    def process_csv_files(directory):
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                with open(file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    data = list(reader)
                    headers = reader.fieldnames

                # Calculate and update 'Taux 2'
                for row in data:
                    for header in headers:
                        if 'Base' in header:
                            rubrique_base = header
                            rubrique_montant_pat = header.replace('Base', 'Montant Pat.')
                            rubrique_taux_2 = header.replace('Base', 'Taux 2')

                            if rubrique_montant_pat in headers and rubrique_taux_2 in headers:
                                base = row.get(rubrique_base)
                                if base == 'NET A PAYER AVANT IMPÔT SUR LE REVENU':
                                    base = ''
                                montant_pat = row.get(rubrique_montant_pat)
                                base_float = convert_to_float(base)
                                montant_pat_float = convert_to_float(montant_pat)
                                if base_float and montant_pat_float and not row.get(rubrique_taux_2):
                                    row[rubrique_taux_2] = round((montant_pat_float / base_float) * 100, 3)

                # Save the updated data back to the CSV file
                output_directory = os.path.join(directory, 'processed')
                os.makedirs(output_directory, exist_ok=True)
                output_file_path = os.path.join(output_directory, filename)
                with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(data)

                #st.write(f"File processed and saved: {output_file_path}")

    process_csv_files(cleaner_output_directory)

    def read_csv(file_path):
        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers_row = next(reader)
            values_rows = list(reader)
        return headers_row, values_rows

    def merge_csv_files(files):
        combined_headers = []
        combined_data = []

        for i, file in enumerate(files):
            headers, rows = read_csv(file)

            if i == 0:
                # Première itération, ajouter toutes les colonnes et les lignes
                combined_headers = headers
                combined_data = rows
            else:
                # Pour les itérations suivantes, fusionner les colonnes similaires et ajouter les nouvelles
                for header in headers:
                    if header not in combined_headers:
                        combined_headers.append(header)

                for row in rows:
                    # Ajouter les données existantes aux colonnes correspondantes
                    new_row = []
                    for header in combined_headers:
                        if header in headers:
                            new_row.append(row[headers.index(header)])
                        else:
                            new_row.append('')

                    combined_data.append(new_row)

        return combined_headers, combined_data

    def add_taux_2_columns(combined_headers, combined_data):
        code_pattern = re.compile(r'^(\d{4})\s')
        headers_to_add = []

        # Trouver les colonnes base et Montant Pat.
        for i, header in enumerate(combined_headers):
            match = code_pattern.match(header)
            if match:
                code = match.group(1)
                base_column = f'{code}Base'
                montant_pat_column = f'{code}Montant Pat.'
                taux_2_column = f'{code} Taux 2'

                #st.write(f"Vérification pour le code: {code}")  # Debug: Affiche le code trouvé

                if base_column in combined_headers and montant_pat_column in combined_headers:
                    #st.write(f"Trouvé: {base_column} et {montant_pat_column}")  # Debug: Affiche les colonnes trouvées
                    base_idx = combined_headers.index(base_column)
                    montant_pat_idx = combined_headers.index(montant_pat_column)

                    # Ajouter la colonne Taux 2 si elle n'existe pas déjà
                    if taux_2_column not in combined_headers:
                        #st.write(f"Ajout de la colonne: {taux_2_column}")  # Debug: Indique qu'une colonne va être ajoutée
                        combined_headers.append(taux_2_column)

                    # Calculer la valeur de Taux 2 pour chaque ligne
                    for row in combined_data:
                        try:
                            base_value = float(row[base_idx])
                            montant_pat_value = float(row[montant_pat_idx])
                            taux_2_value = base_value / montant_pat_value if montant_pat_value != 0 else ''
                        except ValueError:
                            taux_2_value = ''

                        # Vérifie si la colonne Taux 2 est déjà remplie, sinon ajoute la valeur calculée
                        if len(row) < len(combined_headers):
                            row.append(taux_2_value)
                        else:
                            row[combined_headers.index(taux_2_column)] = taux_2_value

                    # Vérification que la colonne a bien été ajoutée

    def write_combined_csv(output_file, combined_headers, combined_data):
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(combined_headers)
            writer.writerows(combined_data)

    def extract_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    input_files = [os.path.join(processed_directory, filename) for filename in os.listdir(processed_directory) if filename.endswith('.csv')]

    # Sort the input files based on the page number
    input_files.sort(key=lambda f: extract_page_number(f))

    # Path to the output CSV file
    output_file = 'combined_output.csv'

    # Merge the CSV files
    combined_headers, combined_data = merge_csv_files(input_files)

    # Add Taux 2 columns
    add_taux_2_columns(combined_headers, combined_data)

    # Write the combined CSV
    write_combined_csv(output_file, combined_headers, combined_data)

    st.write("Combined CSV file with Taux 2 columns has been saved.")

    def update_headers(file_path, output_path):
        # Charger le fichier CSV
        df = pd.read_csv(file_path, header=None)

        # Initialisation des nouveaux noms d'en-tête
        new_headers = []

        # Identifier les en-têtes à partir des valeurs de cellules
        for column in df:
            column_values = df[column].astype(str)  # Convertir toutes les valeurs en chaînes pour la recherche
            if 'Abs.' in column_values.values:
                new_headers.append('Abs.')
            elif 'Date Equipe Hor.Abs.' in column_values.values:
                new_headers.append('Date Equipe Hor.Abs.')
            else:
                new_headers.append(df.columns[column])  # Conserver l'index comme nom de colonne

        # Mettre à jour les en-têtes dans le DataFrame
        df.columns = new_headers

        # Sauvegarder le DataFrame dans un nouveau fichier CSV
        df.to_csv(output_path, index=False)
        #st.write(f"Le fichier a été sauvegardé avec les nouveaux en-têtes : {output_path}")

    for file_path in os.listdir(rest_output_directory) :
        if filename.endswith('.csv'):
            file_path_upd= os.path.join(rest_output_directory,file_path)
            update_headers(file_path_upd,file_path_upd)


    def process_csv_file(file_path):
        absences_par_jour = 0
        absences_par_heure = 0.0

        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            rows = list(reader)

            # Vérifier les colonnes contenant 'Abs.' ou 'Date Equipe Hor.Abs.'
            columns_to_check = set()
            for header in headers:
                if 'Abs.' in header or 'Date Equipe Hor.Abs.' in header:
                    index = headers.index(header)
                    columns_to_check.add(index)

            # Inspecter les colonnes identifiées
            for index in columns_to_check:
                for row in rows:
                    cell = row[index]
                    if isinstance(cell, str) and cell.endswith('AB'):
                        absences_par_jour += 1  # Incrémenter pour chaque 'AB' trouvé
                        # Extraire le nombre avant 'AB'
                        match = re.search(r'(\d+(\.\d+)?)AB$', cell)
                        if match:
                            absences_par_heure += float(match.group(1))  # Ajouter le nombre extrait

        return absences_par_jour, absences_par_heure

    def extract_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')  # Utiliser infini si le numéro de page n'est pas trouvé

    def generate_absences_report(input_directory, output_file):
        report_data = [
            ['Nom du Fichier', 'Absences par Jour', 'Absences par Heure']
        ]

        files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
        files.sort(key=lambda f: extract_page_number(f))  # Trier par numéro de page

        for filename in files:
            file_path = os.path.join(input_directory, filename)
            absences_par_jour, absences_par_heure = process_csv_file(file_path)
            report_data.append([filename, absences_par_jour, absences_par_heure])

        # Sauvegarder le tableau dans un fichier CSV
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(report_data)

        #st.write(f"Rapport d'absences sauvegardé dans {output_file}")


    absence_output_file = 'Absences.csv'

    # Générer le rapport d'absences
    generate_absences_report(rest_output_directory, absence_output_file)

    matricules_file_path = "CSV3/matricules/matricules.csv"
    combined_output_file_path = "combined_output.csv"
    output_file_path = "merged_output.csv"

    # Lire les trois premières matricules depuis le fichier "matricules.csv"
    matricules = []
    with open(matricules_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Ignorer l'en-tête du fichier des matricules
        for i, row in enumerate(reader):
            if i < 298 and row:  # Limiter aux 3 premières lignes non vides
                matricules.append(row[0])
            elif i >= 298 :
                break

    # Lire les données depuis le fichier "combined_output.csv"
    combined_data = []
    with open(combined_output_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        combined_data = [row for row in reader]

    # Vérifier si le nombre de lignes est correct
    if len(matricules) > len(combined_data) - 1:  # Moins 1 pour l'en-tête
        raise ValueError("Le fichier de matricules contient plus de lignes que le fichier combined_output.")

    # Fusionner les matricules en tant que première colonne
    merged_data = []

    # Ajouter l'en-tête de la colonne "Matricule"
    merged_data.append(["Matricule"] + combined_data[0])

    # Ajouter les lignes de données avec les 3 premières matricules
    for i, row in enumerate(combined_data[1:], start=1):
        if i <= len(matricules):  # Limiter à 3 lignes de données
            matricule = matricules[i - 1]
            merged_data.append([matricule] + row)
        else:
            break

    # Écrire le tableau combiné dans un nouveau fichier CSV
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(merged_data)

    st.write(f"Tableau combiné avec les 3 premières matricules enregistré dans {output_file_path}.")

    # Chargement des fichiers
    main_table = pd.read_csv(main_table_path)
    cumul_data = pd.read_excel(uploaded_file_1, engine='openpyxl')
    info_salaries_data = pd.read_excel(uploaded_file_2, engine='openpyxl')
    absences_data = pd.read_csv(absences_path)

    # S'assurer que la colonne 'Matricule' est de type chaîne dans tous les fichiers
    main_table['Matricule'] = main_table['Matricule'].astype(str)
    cumul_data['Matricule'] = cumul_data['Matricule'].astype(str)
    info_salaries_data['Matricule'] = info_salaries_data['Matricule'].astype(str)

    # Fusion des tables en utilisant la colonne 'Matricule'
    merged_df = main_table.merge(cumul_data, on='Matricule', how='left')
    final_df = merged_df.merge(info_salaries_data, on='Matricule', how='left')

    # Déplacer la colonne 'Nom Prénom' à la 2ème position
    cols = list(final_df.columns)
    cols.insert(1, cols.pop(cols.index('Nom Prénom')))
    final_df = final_df[cols]

    # Supprimer la colonne 'Nom'
    if 'Nom' in final_df.columns:
        final_df = final_df.drop(columns=['Nom'])

    # Supprimer la colonne 'Nom du Fichier' du tableau d'absences
    if 'Nom du Fichier' in absences_data.columns:
        absences_data = absences_data.drop(columns=['Nom du Fichier'])

    # Vérifier la correspondance des longueurs et ajuster si nécessaire
    if len(final_df) > len(absences_data):
        # Remplir les lignes manquantes d'absences_data
        absences_data = absences_data.reindex(range(len(final_df))).fillna('')
    elif len(final_df) < len(absences_data):
        # Remplir les lignes manquantes de final_df
        final_df = final_df.reindex(range(len(absences_data))).fillna('')

    # Concaténer horizontalement
    concatenated_df = pd.concat([final_df, absences_data], axis=1)

    # Sauvegarde du fichier final en CSV
    concatenated_df.to_csv(final_output_csv_path, index=False)

    csv_data = concatenated_df.to_csv(index=False).encode('utf-8')

    st.write(f"Fichier sauvegardé sous {final_output_csv_path}")

    st.download_button( label=f"Download fichier restructuré", data= csv_data, file_name="Fichier restrucuturé.csv", mime="text/csv" )

    csv_files_to_delete = glob.glob(os.path.join(output_dir, "**", "*.csv"), recursive=True)

    for csv_file in csv_files_to_delete:
        try:
            os.remove(csv_file)
        except Exception as e:
            st.write(f"Erreur lors de la suppression du fichier {csv_file}: {e}")