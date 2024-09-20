import camelot
import pandas as pd
import os
import re
import csv
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
        return None

required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']
required_elements2 = ['Code','Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux', 'Montant Pat.']

code_libelle_dict = {}
errors = []
rubriques_codes_from_files = set()

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
            st.write(f"Colonne 'Montant Sal.Taux' détectée sur la page {page_number}. Ré-extraction avec les nouveaux paramètres.")
            refined_tables = extract_table_from_pdf(pdf_file_path, edge_tol=500, row_tol=5, pages=str(page_number))

            if refined_tables is not None and len(refined_tables) > 0:
                largest_table = max(refined_tables, key=lambda t: t.df.shape[0] * t.df.shape[1])
                df_stream = largest_table.df
                df_stream.replace('\n', '', regex=True, inplace=True)
                df_stream.fillna('', inplace=True)

        results.append((page_number, df_stream))

    return results

st.title("Extraction de bulletins de paie à partir de PDF")
uploaded_pdf = st.file_uploader("Téléverser un fichier PDF", type=["pdf"])
uploaded_file_1 = st.file_uploader("1er fichier excel", type=['xlsx', 'xls'])
uploaded_file_2 = st.file_uploader("2nd fichier excel", type=['xlsx', 'xls'])

if uploaded_pdf is not None and uploaded_file_1 is not None and uploaded_file_2 is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_pdf.read()) 
        temp_pdf_path = temp_pdf.name 

    output_dir = "CSV3"
    os.makedirs(output_dir, exist_ok=True)

    csv_files = {}

    reader = PdfReader(temp_pdf_path)
    total_pages = len(reader.pages)

    current_page_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = 10 

    max_workers = 4  

    st.write(f"Extraction des tableaux pour toutes les {total_pages} pages...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        other_page_futures = {executor.submit(process_pages, temp_pdf_path, 300, 3, str(page)): page for page in range(1, total_pages + 1)}
        
        tables = []

        for future in concurrent.futures.as_completed(other_page_futures):
            page = other_page_futures[future]
            try:
                results = future.result() 
                for result in results:
                    if isinstance(result, tuple) and len(result) == 2:
                        page_number, df_stream = result
                        if isinstance(df_stream, pd.DataFrame):
                            tables.append(df_stream)
                            csv_files[f"table_page_{page_number}.csv"] = df_stream 

                current_page_count += 1
                progress_value = current_page_count / total_pages
                progress_bar.progress(min(progress_value, 1.0))
                status_text.text(f"Traitement : {min(current_page_count, total_pages)}/{total_pages} pages traitées")
            except Exception as e:
                st.write(f"Erreur lors du traitement de la page {page}: {e}")

    st.write("Extraction des tableaux terminée.")
    st.write('tables : ', len(tables))


    def check_second_line(df, required_elements):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df doit être un DataFrame, mais est de type {type(df)}")
        if len(df) < 2:  
            return False
        
        second_line = df.iloc[0].tolist() 
        if not isinstance(second_line, list):
            raise TypeError(f"second_line doit être une liste, mais est de type {type(second_line)}")
        
        if not all(elem in second_line for elem in required_elements):
            return False
        
        return True




    def split_columns(header, second_line, required_elements):
        if not isinstance(second_line, list):
            raise TypeError(f"second_line doit être une liste, mais est de type {type(second_line)}")

        required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
        other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
        
        return required_indices, other_indices




    if len(tables) == 0:
        st.write("Erreur: La liste input_dfs est vide.")
    else:
        for df in tables:
            if len(df) > 1:
                second_line = df.iloc[0].tolist()

    filtered_dfs = []
    for df in tables:
        if check_second_line(df, required_elements) or check_second_line(df, required_elements2):
            filtered_dfs.append(df)

    st.write("Bulletins enregistrés")

    def check_for_mat(text):
        return 'Mat:' in text

    def extract_matricules(text):
        matricules = []
        for line in text.split('\n'):
            if 'Mat:' in line:
                start = line.find('Mat:') + len('Mat:')
                end = line.find('/ Gest:', start)
                if end == -1:
                    end = len(line)
                matricule = line[start:end].strip()
                if matricule and matricule not in matricules:
                    matricules.append(matricule)
        return matricules

    reader = PdfReader(uploaded_pdf)
    all_matricules = []

    for page in reader.pages:
        text = page.extract_text()
        if check_for_mat(text):
            all_matricules.extend(extract_matricules(text))

    matricules_df = pd.DataFrame(all_matricules, columns=["Matricule"]).drop_duplicates()

    empty_row = pd.DataFrame([[""]], columns=["Matricule"])
    matricules_df = pd.concat([empty_row, matricules_df]).reset_index(drop=True)

    st.write("Matricules distinctes sans doublons :")
    st.write(matricules_df)

    st.write(f"Matricules distinctes enregistrées sans doublons.")


    def rename_second_taux(df):
        if len(df) > 1: 
            second_line = df.iloc[0]  
            taux_indices = [i for i, col in enumerate(second_line) if col == 'Taux']
            if len(taux_indices) > 1:
                df.iloc[1, taux_indices[1]] = 'Taux 2'
        return df



    def check_second_line(df, required_elements):
        if len(df) > 1:  
            second_line = df.iloc[0].tolist() 
            if all(elem in second_line for elem in required_elements):
                return True
        return False


    clean_dfs = []
    rest_dfs = []

    st.write("filtered_dfs :", len(filtered_dfs))
    for df in filtered_dfs:
        header = df.columns.tolist() 
        second_line = df.iloc[0].tolist()

        required_indices, other_indices = split_columns(header, second_line, required_elements + required_elements2)

        clean_df = df.iloc[:, required_indices]
        rest_df = df.iloc[:, other_indices]

        clean_dfs.append(clean_df)
        rest_dfs.append(rest_df)

    st.write("clean_df :",len(clean_dfs))
    st.write("rest_df : ",len(rest_dfs))

    def split_columns(header, second_line, required_elements):
        required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
        other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
        return required_indices, other_indices

    def rename_second_taux(df):
        if len(df) > 1: 
            second_line = df.iloc[0] 
            taux_indices = [i for i, col in enumerate(second_line) if col == 'Taux']
            if len(taux_indices) > 1:
                df.iloc[1, taux_indices[1]] = 'Taux 2'
        return df

    def transform_to_two_lines(df_list, required_elements, required_elements2):
        result = []
        for df in df_list:
            headers_row = []
            values_row = []
            code_libelle_dict = {}
            errors = []
            if not isinstance(df, pd.DataFrame):
                raise ValueError("L'objet passé n'est pas un DataFrame, il s'agit de : " + str(type(df)))
            header = df.columns.tolist()
            second_line = df.iloc[0].tolist() 

            code_index = second_line.index('Code') if 'Code' in second_line else None
            libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
            codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

            rubriques = []
            if codelibelle_index is not None:
                for i, row in df.iterrows():
                    rubrique = row['CodeLibellé']
                    if rubrique and rubrique[0].isdigit():
                        code = rubrique[:4]  
                        libelle = rubrique[5:].strip() if len(rubrique) > 4 else ''
                        if libelle:
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
                for i, row in df.iterrows():
                    rubrique = f"{row['Code']}{row['Libellé']}"
                    if rubrique and rubrique[0].isdigit():
                        code = rubrique[:4]  
                        libelle = row['Libellé'].strip()
                        if libelle:  
                            code_libelle_dict[code] = libelle
                        rubriques.append(rubrique)
                        headers_row.extend([
                            f"{rubrique}Base",
                            f"{rubrique}Taux",
                            f"{rubrique}Montant Sal.",
                            f"{rubrique}Taux 2",
                            f"{rubrique}Montant Pat."
                        ])

            values_row = ['' for _ in range(len(headers_row))]

            for i, row in df.iterrows():
                if codelibelle_index is not None:
                    code_libelle = row['CodeLibellé']
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  
                elif code_index is not None and libelle_index is not None:
                    code_libelle = f"{row['Code']}{row['Libellé']}"
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  
                else:
                    errors.append("Les colonnes 'Code' et 'Libellé' ou 'CodeLibellé' sont manquantes")
                    result.append([headers_row, values_row, code_libelle_dict, errors])  
                    continue

                rubrique_base_header = f"{code_libelle}Base"
                try:
                    rubrique_index = headers_row.index(rubrique_base_header) // 5 * 5  
                    for i, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                        if col in second_line:
                            col_index = second_line.index(col)
                            values_row[rubrique_index + i] = row[col_index]
                except ValueError:
                    errors.append(f"Erreur dans le DataFrame: En-tête '{rubrique_base_header}' introuvable.")
            result.append([headers_row, values_row, code_libelle_dict, errors])

        return result




    result_dfs = []
    code_libelle_dict = {}  
    errors = [] 

    for df in clean_dfs:
        required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']
        required_elements2 = ['Code', 'Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']

        try:
            headers_row, values_row, new_code_libelle_dict, file_errors = transform_to_two_lines(
                df, required_elements, required_elements2)
            code_libelle_dict.update(new_code_libelle_dict)
            errors.extend(file_errors)
            rubriques_codes_from_files = set(new_code_libelle_dict.keys())

            transformed_df = pd.DataFrame([values_row], columns=headers_row)
            result_dfs.append(transformed_df)
        except ValueError as e:
            errors.append(f"Erreur dans le DataFrame: {e}")


    errors_file_path = "errors.txt"
    with open(errors_file_path, mode='w', encoding='utf-8') as errors_file:
        for error in errors:
            errors_file.write(f"{error}\n")

    def generer_taux_2(taux_initial):
        return round(random.uniform(taux_initial - 0.2, taux_initial + 0.2), 2)

    def safe_float_conversion(value):
        if isinstance(value, pd.Series):
            raise TypeError(f"Attendu une valeur unique, mais une série a été reçue : {value}")

        try:
            return float(value)
        except (ValueError, TypeError):
            return None 


    def transform_to_two_lines(df, required_elements, required_elements2):
        headers_row = []
        values_row = []
        if df.empty or len(df) < 2:
            raise ValueError("Le DataFrame est vide ou n'a pas assez de lignes.")
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        header = df.columns.tolist()
        code_index = header.index('Code') if 'Code' in header else None
        libelle_index = header.index('Libellé') if 'Libellé' in header else None
        codelibelle_index = header.index('CodeLibellé') if 'CodeLibellé' in header else None

        if code_index is None and libelle_index is None and codelibelle_index is None:
            raise ValueError("Les colonnes 'Code', 'Libellé', et 'CodeLibellé' sont manquantes.")

        rubriques = []
        if codelibelle_index is not None:
            for i, row in df.iterrows():
                rubrique = row[codelibelle_index]
                if rubrique not in rubriques:
                    if rubrique and rubrique[0].isdigit():
                        rubrique = rubrique[:4] 
                    rubriques.append(rubrique)
                    headers_row.extend([
                        f"{rubrique}Base",
                        f"{rubrique}Taux",
                        f"{rubrique}Montant Sal.",
                        f"{rubrique}Taux 2",  
                        f"{rubrique}Montant Pat."
                    ])
        elif code_index is not None and libelle_index is not None:
            for i, row in df.iterrows():
                rubrique = f"{row[code_index]}{row[libelle_index]}"
                if rubrique not in rubriques:
                    if rubrique and rubrique[0].isdigit():
                        rubrique = rubrique[:4]  
                    rubriques.append(rubrique)
                    headers_row.extend([
                        f"{rubrique}Base",
                        f"{rubrique}Taux",
                        f"{rubrique}Montant Sal.",
                        f"{rubrique}Taux 2", 
                        f"{rubrique}Montant Pat."
                    ])

        values_row = ['' for _ in range(len(headers_row))]

        for i, row in df.iterrows():
            if codelibelle_index is not None:
                code_libelle = row[codelibelle_index]
                if code_libelle and code_libelle[0].isdigit():
                    code_libelle = code_libelle[:4] 
            elif code_index is not None and libelle_index is not None:
                code_libelle = f"{row[code_index]}{row[libelle_index]}"
                if code_libelle and code_libelle[0].isdigit():
                    code_libelle = code_libelle[:4]
            else:
                raise ValueError("Les colonnes 'Code' et 'Libellé' ou 'CodeLibellé' sont manquantes")

            rubrique_base_header = f"{code_libelle}Base"
            try:
                rubrique_index = headers_row.index(rubrique_base_header) // 5 * 5 
                for j, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                    if col in header:
                        col_index = header.index(col)
                        values_row[rubrique_index + j] = row[col_index]

                taux_value = safe_float_conversion(values_row[rubrique_index + 1])  
                if taux_value is not None: 
                    taux_2_value = generer_taux_2(taux_value) 
                    values_row[rubrique_index + 3] = taux_2_value 

            except ValueError:
                raise ValueError(f"Erreur dans le DataFrame: En-tête '{rubrique_base_header}' introuvable.")

        return [headers_row, values_row]

    
    transformed_dfs =  []

    for df in clean_dfs:
        try:
            headers_row, values_row = transform_to_two_lines(df, required_elements, required_elements2)
            transformed_df = pd.DataFrame([values_row], columns=headers_row)
            transformed_dfs.append(transformed_df)
        except ValueError as e:
            
            errors.append(f"Erreur: {e}")
            st.write(f"Erreur lors de la transformation de df : {e}")

    st.write("DataFrames transformés et stockés.")

    st.write(transformed_dfs[1].head())
    


    def convert_to_float(value):
        if value:
            try:
                value = value.replace('.', '').replace(',', '.')
                return float(value)
            except ValueError:
                return value
        return value

    def process_dataframes(dfs):
        processed_dfs = []

        for df in dfs:
            headers = df.columns.tolist()
            for index, row in df.iterrows():
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

                            if base_float and montant_pat_float and pd.isna(row.get(rubrique_taux_2)):
                                df.at[index, rubrique_taux_2] = round((montant_pat_float / base_float) * 100, 3)

            processed_dfs.append(df)

        return processed_dfs


    def read_csv(file_path):
        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers_row = next(reader)
            values_rows = list(reader)
        return headers_row, values_rows

    def merge_dataframes(dfs):
        combined_headers = []
        combined_data = []

        for i, df in enumerate(dfs):
            headers = df.columns.tolist()
            rows = df.values.tolist()

            if i == 0:
                combined_headers = headers
                combined_data = rows
            else:
                for header in headers:
                    if header not in combined_headers:
                        combined_headers.append(header)

                for row in rows:
                    new_row = []
                    for header in combined_headers:
                        if header in headers:
                            new_row.append(row[headers.index(header)])
                        else:
                            new_row.append('')

                    combined_data.append(new_row)

        merged_df = pd.DataFrame(combined_data, columns=combined_headers)
        return merged_df


    def add_taux_2_columns(df):

        code_pattern = re.compile(r'^(\d{4})')

        for header in df.columns:
            header_str = str(header)
            match = code_pattern.match(header_str)
        
            st.write(f"Processing header: {header_str}")

            if match:
                code = match.group(1)
                base_column = f'{code}Base'
                montant_pat_column = f'{code}Montant Pat.'
                taux_2_column = f'{code}Taux 2'

                if base_column in df.columns and montant_pat_column in df.columns:
                    df[base_column] = df[base_column].astype(str).str.replace('.', '').str.replace(',', '.').str.strip()
                    df[montant_pat_column] = df[montant_pat_column].astype(str).str.replace('.', '').str.replace(',', '.').str.strip()
                    df[base_column] = pd.to_numeric(df[base_column], errors='coerce')
                    df[montant_pat_column] = pd.to_numeric(df[montant_pat_column], errors='coerce')

                    if taux_2_column not in df.columns:
                        df[taux_2_column] = None

                    df[taux_2_column] = df.apply(
                        lambda row: row[base_column] / row[montant_pat_column]
                        if pd.notna(row[base_column]) and pd.notna(row[montant_pat_column]) and row[montant_pat_column] != 0
                        else None,
                        axis=1
                    )

        return df


    def write_combined_csv(output_file, combined_headers, combined_data):
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(combined_headers)
            writer.writerows(combined_data)

    def extract_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')


    input_dfs = transformed_dfs

    combined_df = merge_dataframes(input_dfs)

    combined_df = add_taux_2_columns(combined_df)

    

    st.write("Combined CSV file with Taux 2 columns has been saved.")

    def update_headers(df):
        new_headers = []

        for column in df.columns:
            column_values = df[column].astype(str)
            if 'Abs.' in column_values.values:
                new_headers.append('Abs.')
            elif 'Date Equipe Hor.Abs.' in column_values.values:
                new_headers.append('Date Equipe Hor.Abs.')
            else:
                new_headers.append(column)
        df.columns = new_headers
        return df

    def process_dataframe(df):
        absences_par_jour = 0
        absences_par_heure = 0.0

        columns_to_check = [col for col in df.columns if 'Abs.' in str(col) or 'Date Equipe Hor.Abs.' in str(col)]

        for col in columns_to_check:
            for cell in df[col].astype(str):
                if cell.endswith('AB'):
                    absences_par_jour += 1  
                    match = re.search(r'(\d+(\.\d+)?)AB$', cell)
                    if match:
                        absences_par_heure += float(match.group(1))

        return absences_par_jour, absences_par_heure

    def generate_absences_report(dfs):
        report_data = [
            ['Nom du fichier', 'Absences par Jour', 'Absences par Heure']
        ]

        for i, df in enumerate(dfs):
            absences_par_jour, absences_par_heure = process_dataframe(df)
            report_data.append([f"DataFrame_{i+1}", absences_par_jour, absences_par_heure])
        report_df = pd.DataFrame(report_data[1:], columns=report_data[0])
        return report_df

    def get_matricules(df, limit=600):
        return df.iloc[:limit, 0].tolist()

    def merge_matricules_with_combined(matricules, combined_df):
        if len(matricules) - 2 > len(combined_df):
            st.write(len(matricules) - 2)
            st.write(len(combined_df))
            raise ValueError("Le DataFrame de matricules contient plus de lignes que le DataFrame combined_output.")

        matricules_df = pd.DataFrame(matricules[:len(combined_df)], columns=['Matricule'])
        combined_df = pd.concat([matricules_df, combined_df.reset_index(drop=True)], axis=1)

        return combined_df
    matricules = get_matricules(matricules_df, limit=600)

    def clean_matricule_column(df, column_name='Matricule'):
        df[column_name] = df[column_name].astype(str).str.replace('\n', '').str.strip()
        return df

    

    if len(matricules) -2 > len(combined_df):
        st.write(f"Nombre de matricules : {len(matricules) - 2}")
        st.write(f"Nombre de lignes dans combined_df : {len(combined_df)}")
        raise ValueError("Le DataFrame de matricules contient plus de lignes que le DataFrame combined_output.")
    
    info_salaries_data = pd.read_excel(uploaded_file_1, engine='openpyxl')
    info_salaries_data['Matricule'] = info_salaries_data['Matricule'].astype(str)
    cumul_data = pd.read_excel(uploaded_file_2, engine='openpyxl')
    cumul_data['Matricule'] = cumul_data['Matricule'].astype(str)
    

    merged_df = merge_matricules_with_combined(matricules, combined_df)

    merged_df = clean_matricule_column(merged_df, 'Matricule')
    info_salaries_data = clean_matricule_column(info_salaries_data, 'Matricule')
    cumul_data = clean_matricule_column(cumul_data, 'Matricule')

    merged_df['Matricule'] = merged_df['Matricule'].astype(str)
    inter_df = merged_df.merge(cumul_data, on='Matricule', how='left')
    inter_df['Matricule'] = inter_df['Matricule'].astype(str)
    info_salaries_data['Matricule'] = info_salaries_data['Matricule'].astype(str)
    final_df = inter_df.merge(info_salaries_data, on='Matricule', how='left')

    st.write(f"Tableau combiné avec les 3 premières matricules.")

    absences_data = generate_absences_report(rest_dfs)

    cols = list(final_df.columns)

    if 'Nom Prénom' in cols:
        cols.insert(1, cols.pop(cols.index('Nom Prénom')))

    final_df = final_df[cols]

    if 'Nom' in final_df.columns:
        final_df = final_df.drop(columns=['Nom'])

    

    if len(final_df) > len(absences_data):
        absences_data = absences_data.reindex(range(len(final_df))).fillna('')
    elif len(final_df) < len(absences_data):
        final_df = final_df.reindex(range(len(absences_data))).fillna('')

    concatenated_df = pd.concat([final_df, absences_data], axis=1)

    csv_data = concatenated_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Télécharger le fichier restructuré",
        data=csv_data,
        file_name="Fichier_restructuré.csv",
        mime="text/csv"
    )
