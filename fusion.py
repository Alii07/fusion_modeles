import streamlit as st
import pandas as pd
import camelot
import os
import csv
import shutil
from PyPDF2 import PdfReader
import random
import re

required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']
required_elements2 = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']

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

    print(f"Rapport d'absences sauvegardé dans {output_file}")

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
    print(f"Le fichier a été sauvegardé avec les nouveaux en-têtes : {output_path}")

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

            print(f"Vérification pour le code: {code}")  # Debug: Affiche le code trouvé

            if base_column in combined_headers and montant_pat_column in combined_headers:
                print(f"Trouvé: {base_column} et {montant_pat_column}")  # Debug: Affiche les colonnes trouvées
                base_idx = combined_headers.index(base_column)
                montant_pat_idx = combined_headers.index(montant_pat_column)

                # Ajouter la colonne Taux 2 si elle n'existe pas déjà
                if taux_2_column not in combined_headers:
                    print(f"Ajout de la colonne: {taux_2_column}")  # Debug: Indique qu'une colonne va être ajoutée
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
                if taux_2_column in combined_headers:
                    print(f"La colonne '{taux_2_column}' a été ajoutée avec succès.")
                else:
                    print(f"Erreur : La colonne '{taux_2_column}' n'a pas été ajoutée.")

def write_combined_csv(output_file, combined_headers, combined_data):
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(combined_headers)
        writer.writerows(combined_data)

def extract_page_number(filename):
    match = re.search(r'_page_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')  # Utiliser infini si pas de numéro de page

def transform_to_single_line(file_path, required_elements, required_elements2):
    combined_headers_row = []
    values_row = []

    with open(file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Lire la première ligne (en-tête)
        second_line = next(reader)  # Lire la deuxième ligne (colonnes requises)

        # Vérifier la présence des colonnes 'Code' et 'Libellé'
        code_index = second_line.index('Code') if 'Code' in second_line else None
        libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
        codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

        # Rewind the reader to read the file again for values
        infile.seek(0)
        next(reader)  # Ignore the first line (header)
        next(reader)  # Ignore the second line (column names)

        # Remplir les lignes avec les données correspondantes
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

            # Ajouter les rubriques aux en-têtes combinés
            for col in ['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']:
                if col in second_line:
                    combined_headers_row.append(f"{code_libelle} {col}")
                    col_index = second_line.index(col)
                    values_row.append(row[col_index])
                elif col == 'Taux 2':
                    # Calculer "XXXX Taux 2" s'il n'existe pas
                    base_index = second_line.index('Base') if 'Base' in second_line else None
                    montant_pat_index = second_line.index('Montant Pat.') if 'Montant Pat.' in second_line else None

                    if base_index is not None and montant_pat_index is not None:
                        base_value = float(row[base_index]) if row[base_index] else None
                        montant_pat_value = float(row[montant_pat_index]) if row[montant_pat_index] else None

                        if base_value and montant_pat_value:
                            taux2_value = base_value / montant_pat_value if montant_pat_value != 0 else None
                            combined_headers_row.append(f"{code_libelle} Taux 2")
                            values_row.append(taux2_value)

            # Ajouter le reste des valeurs
            if 'Montant Pat.' in second_line:
                combined_headers_row.append(f"{code_libelle} Montant Pat.")
                montant_pat_index = second_line.index('Montant Pat.')
                values_row.append(row[montant_pat_index])

    return combined_headers_row, values_row


def extract_table_from_pdf(pdf_path, edge_tol=500, row_tol=5, pages='all'):
    try:
        # Extraire les tableaux à partir du PDF en utilisant la méthode 'stream'
        tables = camelot.read_pdf(pdf_path, flavor='stream', edge_tol=edge_tol, row_tol=row_tol, pages=pages)

        # Initialiser une liste pour stocker les DataFrames extraits de chaque page
        extracted_tables = []

        # Parcourir les tables extraites
        for table in tables:
            # Extraire le DataFrame de la table
            df = table.df

            # Remplacer les sauts de ligne (\n) par une chaîne vide
            df.replace('\n', '', regex=True, inplace=True)

            # Remplacer les valeurs NaN par une chaîne vide
            df.fillna('', inplace=True)

            # Parcourir chaque cellule du DataFrame
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    cell_content = df.iloc[i, j]
                    # Vérifier si la cellule contient un numéro suivi de deux lettres
                    if isinstance(cell_content, str) and len(cell_content) >= 3 and cell_content[0].isdigit() and cell_content[1].isalpha() and cell_content[2].isalpha():
                        # Cas 1: Si la cellule suivante est vide, déplacer son contenu
                        if j < len(df.columns) - 1 and df.iloc[i, j+1] == '':
                            df.iloc[i, j+1] = cell_content[3:]
                        # Cas 2: Si la cellule suivante est remplie, créer une nouvelle colonne
                        elif j < len(df.columns) - 1:
                            df.insert(j+1, f'Colonne{j+1}_new', '')
                            df.iloc[i, j+1] = cell_content[3:]

            # Ajouter le DataFrame à la liste des tableaux extraits
            extracted_tables.append(df)

        return extracted_tables

    except Exception as e:
        print(f"Erreur lors de l'extraction des tableaux : {e}")
        return None

def check_for_mat(text):
    return 'Mat:' in text

# Fonction pour extraire les matricules du texte
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
    
def transform_and_combine_tables(extracted_tables):
    # Initialiser une liste pour les lignes transformées
    transformed_data = []

    # Définir les colonnes de sortie
    columns = [
        'CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux Montant Pat.',
        'Du', 'Date', 'Equipe', 'Hor.', 'Abs.'
    ]

    # Parcourir chaque DataFrame extrait
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

    # Créer un DataFrame combiné à partir des lignes transformées
    combined_df = pd.DataFrame(transformed_data, columns=columns)
    return combined_df


def save_table_to_csv(df, file_path):
    try:
        # Convertir le DataFrame en texte CSV
        table_text = df.to_csv(index=False, header=False, encoding='utf-8-sig')

        # Enregistrer le texte dans un fichier CSV
        df.to_csv(file_path, index=False, header=False, encoding='utf-8-sig')

        print("Le texte extrait a été enregistré dans le fichier CSV avec succès.")
    except Exception as e:
        print("Erreur lors de l'enregistrement du texte dans le fichier CSV:", str(e))


def save_text_to_txt(text, file_path):
    try:
        # Enregistrer le texte dans un fichier texte
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
        print("Le texte extrait a été enregistré dans le fichier TXT avec succès.")
    except Exception as e:
        print("Erreur lors de l'enregistrement du texte dans le fichier TXT:", str(e))

def clean_text(text):
    # Remplacer les valeurs NaN par une chaîne vide
    cleaned_text = text.replace('NaN', '')

    # Supprimer les sauts de ligne
    cleaned_text = cleaned_text.replace('\n', ' ')

    return cleaned_text

def display_table_head(df, num_rows=30):
    # Afficher les premières lignes du DataFrame sous forme de tableau
    print("Quelques lignes du DataFrame extrait :")
    print(df.head(num_rows))


def check_second_line(file_path, required_elements):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Ignore la première ligne
        second_line = next(reader, None)  # Lire la deuxième ligne
        if second_line and all(elem in second_line for elem in required_elements):
            return True
    return False

def check_second_line(file_path, required_elements):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Ignore la première ligne
        second_line = next(reader, None)  # Lire la deuxième ligne
        if second_line and all(elem in second_line for elem in required_elements):
            return True
    return False

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

            print(f"File processed and saved: {output_file_path}")

# Fonction pour générer un taux selon une certaine logique
def generer_taux_2(taux_initial):
    return round(random.uniform(taux_initial - 0.2, taux_initial + 0.2), 2)

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

            # Générer et ajouter la valeur pour 'Taux 2'
            taux_2_value = generer_taux_2(float(values_row[rubrique_index + 1]))  # 'Taux 2' basé sur 'Taux'
            values_row[rubrique_index + 3] = taux_2_value  # Placer 'Taux 2' dans la bonne colonne

    return [headers_row, values_row]

filtered_files = []
csv_directory = "CSV3"
output_directory = os.path.join(csv_directory, "bulletins")

# Créer le sous-répertoire si nécessaire
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        if check_second_line(file_path, required_elements) or check_second_line(file_path, required_elements2):
            filtered_files.append(file_path)

# Copier les fichiers CSV filtrés dans le sous-répertoire
for file in filtered_files:
    shutil.copy(file, output_directory)





st.title("Détection d'anomalies dans les cotisations URSSAF - Continental")

pdf_upload = st.file_uploader("Entrez votre bulletin de paie (Format pdf)", type = ['pdf'])

if pdf_upload :
    
    pdf_path = pdf_upload

    # Pages spécifiques où edge_tol=500 et row_tol=5
    specific_pages = [81, 95, 101, 229, 365, 405, 417, 431, 446]
    specific_pages_str = ','.join(map(str, specific_pages))

    # Extraire les tableaux pour les pages spécifiques
    tables_stream = extract_table_from_pdf(pdf_path, edge_tol=500, row_tol=5, pages=specific_pages_str)

    if tables_stream is not None:
        for i, table in enumerate(tables_stream):
            page_number = table.parsing_report['page']

            # Extraire le DataFrame
            df_stream = table.df
            df_stream.replace('\n', '', regex=True, inplace=True)
            df_stream.fillna('', inplace=True)

            # Sauvegarder le DataFrame extrait
            stream_output_csv_file = f"CSV3/_tableau_page_{page_number}.csv"
            save_table_to_csv(df_stream, stream_output_csv_file)

    # Extraire les tableaux pour le reste des pages avec edge_tol=300 et row_tol=3
    tables_stream = extract_table_from_pdf(pdf_path, edge_tol=300, row_tol=3, pages='1-end')

    if tables_stream is not None:
        for i, table in enumerate(tables_stream):
            page_number = table.parsing_report['page']

            # Ignorer les pages spécifiques déjà traitées
            if page_number in specific_pages:
                continue

            # Extraire le DataFrame
            df_stream = table.df
            df_stream.replace('\n', '', regex=True, inplace=True)
            df_stream.fillna('', inplace=True)

            # Sauvegarder le DataFrame extrait
            stream_output_csv_file = f"CSV3/_tableau_page_{page_number}.csv"
            save_table_to_csv(df_stream, stream_output_csv_file)
    
    reader = PdfReader(pdf_upload)
    all_matricules = set()

    for page in reader.pages:
        text = page.extract_text()
        if check_for_mat(text):
            all_matricules.update(extract_matricules(text))

    # Créer le répertoire pour stocker les matricules
    csv_directory = "CSV3"
    output_directory = os.path.join(csv_directory, "matricules")
    os.makedirs(output_directory, exist_ok=True)

    # Enregistrer les matricules distinctes dans un nouveau fichier CSV
    matricules_file_path = os.path.join(output_directory, "matricules.csv")
    with open(matricules_file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Matricule"])
        writer.writerow([])  # Ajouter une ligne vide après l'en-tête
        for matricule in sorted(all_matricules):
            writer.writerow([matricule])
    filtered_files = []
    csv_directory = "CSV3"
    output_directory = os.path.join(csv_directory, "bulletins")
    clean_output_directory = os.path.join(output_directory, "bulletins_propres")
    rest_output_directory = os.path.join(output_directory, "restes_tableaux")

    # Créer les sous-répertoires si nécessaire
    os.makedirs(clean_output_directory, exist_ok=True)
    os.makedirs(rest_output_directory, exist_ok=True)

    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_directory, filename)
            if check_second_line(file_path, required_elements) or check_second_line(file_path, required_elements2):
                rename_second_taux(file_path)
                filtered_files.append(file_path)

    # Diviser les tableaux et sauvegarder les résultats dans les répertoires appropriés
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
            
    cleaner_output_directory = os.path.join(output_directory, "bulletins_propres_structurés")

    # Créer les sous-répertoires si nécessaire
    os.makedirs(cleaner_output_directory, exist_ok=True)

    # Dictionnaire pour stocker les correspondances code rubrique : libellé
    code_libelle_dict = {}
    errors = []
    rubriques_codes_from_files = set()  # Set pour stocker tous les codes de rubriques des fichiers

    # Diviser les tableaux et sauvegarder les résultats dans les répertoires appropriés
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

    # Trier le dictionnaire par les codes rubriques
    sorted_code_libelle = sorted(code_libelle_dict.items())

    # Écrire le dictionnaire trié dans un fichier texte
    dictionnaire_file_path = "Dictionnaire.txt"
    with open(dictionnaire_file_path, mode='w', encoding='utf-8') as dict_file:
        for code, libelle in sorted_code_libelle:
            dict_file.write(f"{code} : {libelle}\n")

    # Charger les codes du dictionnaire final
    final_codes = set()
    with open(dictionnaire_file_path, mode='r', encoding='utf-8') as dict_file:
        for line in dict_file:
            code = line.split(':')[0].strip()
            final_codes.add(code)

    # Extraire les codes de rubriques qui ne sont pas dans le dictionnaire
    missing_codes = rubriques_codes_from_files - final_codes

    # Imprimer les codes manquants
    print("Codes de rubriques manquants dans le dictionnaire :")
    for code in missing_codes:
        print(code)

    # Écrire les erreurs dans un fichier texte
    errors_file_path = "errors.txt"
    with open(errors_file_path, mode='w', encoding='utf-8') as errors_file:
        for error in errors:
            errors_file.write(f"{error}\n")

    os.makedirs(clean_output_directory, exist_ok=True)

# Diviser les tableaux et sauvegarder les résultats dans les répertoires appropriés
for filename in os.listdir(clean_output_directory):
    if filename.endswith(".csv") and filename != "bulletins_propres":
        file_path = os.path.join(clean_output_directory, filename)

        try:
            headers_row, values_row = transform_to_two_lines(file_path, required_elements, required_elements2)

            # Sauvegarder le fichier transformé
            clean_file_path = os.path.join(cleaner_output_directory, "restructured_" + filename)
            with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                writer = csv.writer(clean_outfile)
                writer.writerow(headers_row)
                writer.writerow(values_row)
        except ValueError as e:
            pass

    process_csv_files(cleaner_output_directory)

    for filename in os.listdir(clean_output_directory):
        if filename.endswith(".csv") and filename != "bulletins_propres":
            file_path = os.path.join(clean_output_directory, filename)

            # Transformer les données en une seule ligne
            required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']
            required_elements2 = ['Code', 'Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']

            try:
                combined_headers_row, values_row = transform_to_single_line(file_path, required_elements, required_elements2)

                # Sauvegarder le fichier transformé
                clean_file_path = os.path.join(cleaner_output_directory, "restructured_" + filename)
                with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                    writer = csv.writer(clean_outfile)
                    writer.writerow(combined_headers_row)
                    writer.writerow(values_row)
            except ValueError as e:
                pass
    
    input_directory = 'CSV3/bulletins/bulletins_propres/bulletins_propres_structurés/processed'

    # Paths to the input CSV files
    input_files = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory) if filename.endswith('.csv')]

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

    print("Combined CSV file with Taux 2 columns has been saved.")

    input_directory = 'CSV3/bulletins/restes_tableaux'

    for file_path in os.listdir(input_directory) :
        if filename.endswith('.csv'):
            file_path_upd= os.path.join(input_directory,file_path)
            update_headers(file_path_upd,file_path_upd)

    output_file = 'Absences.csv'

    # Générer le rapport d'absences
    generate_absences_report(input_directory, output_file)

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

    main_table_path = 'merged_output.csv'
    cumul_path = '/content/drive/MyDrive/Cumul de janvier à juin.xlsx'
    info_salaries_path = '/content/drive/MyDrive/Information sur les salariés.xlsx'
    absences_path = 'Absences.csv'
    output_csv_path = 'final_output.csv'

    # Chargement des fichiers
    main_table = pd.read_csv(main_table_path)
    cumul_data = pd.read_excel(cumul_path)
    info_salaries_data = pd.read_excel(info_salaries_path)
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
    concatenated_df.to_csv(output_csv_path, index=False)

    print(f"Fichier sauvegardé sous {output_csv_path}")

    st.download_button("Télécharger le fichier structuré :",output_csv_path, mime="table/csv" )




    