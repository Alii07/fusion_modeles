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
from io import StringIO

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
            print(f"Colonne 'Montant Sal.Taux' détectée sur la page {page_number}. Ré-extraction avec les nouveaux paramètres.")

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
        
        for future in concurrent.futures.as_completed(other_page_futures):
            page = other_page_futures[future]
            try:
                results = future.result()
                for page_number, df_stream in results:
                    stream_output_csv_file = os.path.join(output_dir, f"table_page_{page_number}.csv")
                    csv_content = save_table_to_memory_csv(df_stream)
                    csv_files[f"table_page_{page_number}.csv"] = df_stream
                
                current_page_count += 1
                progress_value = current_page_count / total_pages
                if progress_value > 1.0:
                    progress_value = 1.0  
                progress_bar.progress(progress_value)
                status_text.text(f"Traitement : {min(current_page_count, total_pages)}/{total_pages} pages traitées")
                
            except Exception as e:
                st.write(f"Erreur lors du traitement des pages {page}: {e}")

    st.write("Extraction des tableaux terminée.")


    def check_second_line(file_path, required_elements):
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  
            second_line = next(reader, None)  
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


    for file in filtered_files:
        shutil.copy(file, output_directory)

    st.write("Fichiers CSV filtrés enregistrés dans CSV3/bulletins.")

    def check_for_mat(text):
        return 'Mat:' in text

    def extract_matricules(text):
        matricules = set()
        for line in text.split('\n'):
            if 'Mat:' in line:
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
        writer.writerow([])  
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
            next(reader)  
            second_line = next(reader, None) 
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
            header = next(reader)  
            second_line = next(reader)  
            required_indices, other_indices = split_columns(header, second_line, required_elements + required_elements2)

            clean_file_path = os.path.join(clean_output_directory, os.path.basename(file))
            rest_file_path = os.path.join(rest_output_directory, os.path.basename(file))

            with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                writer = csv.writer(clean_outfile)
                writer.writerow([header[i] for i in required_indices])
                writer.writerow([second_line[i] for i in required_indices])
                for row in reader:
                    writer.writerow([row[i] for i in required_indices])

            infile.seek(0)
            reader = csv.reader(infile)
            header = next(reader)  
            second_line = next(reader)  

            with open(rest_file_path, mode='w', newline='', encoding='utf-8') as rest_outfile:
                writer = csv.writer(rest_outfile)
                writer.writerow([header[i] for i in other_indices])
                writer.writerow([second_line[i] for i in other_indices])
                for row in reader:
                    writer.writerow([row[i] for i in other_indices])

    st.write("Fichiers CSV divisés et sauvegardés dans les répertoires appropriés.")

    def split_columns(header, second_line, required_elements):
        required_indices = [i for i, col in enumerate(second_line) if col in required_elements]
        other_indices = [i for i, col in enumerate(second_line) if col not in required_elements]
        return required_indices, other_indices

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
            header = next(reader) 
            second_line = next(reader)  

            code_index = second_line.index('Code') if 'Code' in second_line else None
            libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
            codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

            rubriques = []
            if codelibelle_index is not None:
                for row in reader:
                    rubrique = row[codelibelle_index]
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
                for row in reader:
                    rubrique = f"{row[code_index]}{row[libelle_index]}"
                    if rubrique and rubrique[0].isdigit():
                        code = rubrique[:4]  
                        libelle = row[libelle_index].strip()
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

            infile.seek(0)
            next(reader) 
            next(reader)  

            values_row = ['' for _ in range(len(headers_row))]

            
            for row in reader:
                if codelibelle_index is not None:
                    code_libelle = row[codelibelle_index]
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  
                elif code_index is not None and libelle_index is not None:
                    code_libelle = f"{row[code_index]}{row[libelle_index]}"
                    if code_libelle and code_libelle[0].isdigit():
                        code_libelle = code_libelle[:4]  
                else:
                    errors.append("Les colonnes 'Code' et 'Libellé' ou 'CodeLibellé' sont manquantes")
                    return [headers_row, values_row, code_libelle_dict, errors]  

                rubrique_base_header = f"{code_libelle}Base"
                try:
                    rubrique_index = headers_row.index(rubrique_base_header) // 5 * 5  
                    for i, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                        if col in second_line:
                            col_index = second_line.index(col)
                            values_row[rubrique_index + i] = row[col_index]
                except ValueError:
                    errors.append(f"Erreur dans le fichier {file_path}: En-tête '{rubrique_base_header}' introuvable.")

        return [headers_row, values_row, code_libelle_dict, errors]



    for filename in os.listdir(clean_output_directory):
        if filename.endswith(".csv") and filename != "bulletins_propres":
            file_path = os.path.join(clean_output_directory, filename)

            required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']
            required_elements2 = ['Code', 'Libellé', 'Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']

            try:
                headers_row, values_row, new_code_libelle_dict, file_errors = transform_to_two_lines(
                    file_path, required_elements, required_elements2)
                code_libelle_dict.update(new_code_libelle_dict)
                errors.extend(file_errors)

                rubriques_codes_from_files.update(new_code_libelle_dict.keys())

                clean_file_path = os.path.join(cleaner_output_directory, "restructured_" + filename)
                with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                    writer = csv.writer(clean_outfile)
                    writer.writerow(headers_row)
                    writer.writerow(values_row)
            except ValueError as e:
                errors.append(f"Erreur dans le fichier {filename}: {e}")

    sorted_code_libelle = sorted(code_libelle_dict.items())

    dictionnaire_file_path = "./Dictionnaire2.txt"
    with open(dictionnaire_file_path, mode='w', encoding='utf-8') as dict_file:
        for code, libelle in sorted_code_libelle:
            dict_file.write(f"{code} : {libelle}\n")

    final_codes = set()
    with open(dictionnaire_file_path, mode='r', encoding='utf-8') as dict_file:
        for line in dict_file:
            code = line.split(':')[0].strip()
            final_codes.add(code)

    missing_codes = rubriques_codes_from_files - final_codes

    errors_file_path = "errors.txt"
    with open(errors_file_path, mode='w', encoding='utf-8') as errors_file:
        for error in errors:
            errors_file.write(f"{error}\n")
    def generer_taux_2(taux_initial):
        return round(random.uniform(taux_initial - 0.2, taux_initial + 0.2), 2)

    def safe_float_conversion(value):
        try:
            return float(value)
        except ValueError:
            return None  
    
    def transform_to_two_lines(file_path, required_elements, required_elements2):
        headers_row = []
        values_row = []

        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  
            second_line = next(reader)

            code_index = second_line.index('Code') if 'Code' in second_line else None
            libelle_index = second_line.index('Libellé') if 'Libellé' in second_line else None
            codelibelle_index = second_line.index('CodeLibellé') if 'CodeLibellé' in second_line else None

            rubriques = []
            if codelibelle_index is not None:
                for row in reader:
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
                for row in reader:
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

            infile.seek(0)
            next(reader)  
            next(reader)  

           
            values_row = ['' for _ in range(len(headers_row))]

            
            for row in reader:
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

                rubrique_index = headers_row.index(f"{code_libelle}Base") // 5 * 5  
                for i, col in enumerate(['Base', 'Taux', 'Montant Sal.', 'Taux 2', 'Montant Pat.']):
                    if col in second_line:
                        col_index = second_line.index(col)
                        values_row[rubrique_index + i] = row[col_index]

                
                taux_value = safe_float_conversion(values_row[rubrique_index + 1])  
                if taux_value is not None:  
                    taux_2_value = generer_taux_2(taux_value)  
                    values_row[rubrique_index + 3] = taux_2_value  

        return [headers_row, values_row]

    for filename in os.listdir(clean_output_directory):
        if filename.endswith(".csv") and filename != "bulletins_propres":
            file_path = os.path.join(clean_output_directory, filename)
            required_elements = ['CodeLibellé', 'Base', 'Taux', 'Montant Sal.', 'Montant Pat.']
            required_elements2 = ['Code', 'Libellé', 'Base', 'Taux', 'Montant Sal.', 'Montant Pat.']

            try:
                headers_row, values_row = transform_to_two_lines(file_path, required_elements, required_elements2)

                clean_file_path = os.path.join(cleaner_output_directory, "restructured_" + filename)
                with open(clean_file_path, mode='w', newline='', encoding='utf-8') as clean_outfile:
                    writer = csv.writer(clean_outfile)
                    writer.writerow(headers_row)
                    writer.writerow(values_row)
                    
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

                output_directory = os.path.join(directory, 'processed')
                os.makedirs(output_directory, exist_ok=True)
                output_file_path = os.path.join(output_directory, filename)
                with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(data)

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

        return combined_headers, combined_data

    def add_taux_2_columns(combined_headers, combined_data):
        code_pattern = re.compile(r'^(\d{4})\s')

        for i, header in enumerate(combined_headers):
            match = code_pattern.match(header)
            if match:
                code = match.group(1)
                base_column = f'{code}Base'
                montant_pat_column = f'{code}Montant Pat.'
                taux_2_column = f'{code} Taux 2'

                if base_column in combined_headers and montant_pat_column in combined_headers:
                    base_idx = combined_headers.index(base_column)
                    montant_pat_idx = combined_headers.index(montant_pat_column)

                    if taux_2_column not in combined_headers:
                        combined_headers.append(taux_2_column)

                    for row in combined_data:
                        try:
                            base_value = float(row[base_idx])
                            montant_pat_value = float(row[montant_pat_idx])
                            taux_2_value = base_value / montant_pat_value if montant_pat_value != 0 else ''

                            if taux_2_value != '':
                                if code == '7C00' :
                                    taux_2_value = format(round(taux_2_value, 2), '.2f')
                                else :
                                    taux_2_value = format(round(taux_2_value, 2), '.3f')
                        except (ValueError, TypeError): 
                            taux_2_value = ''

                        if len(row) < len(combined_headers):
                            row.append(taux_2_value)
                        else:
                            row[combined_headers.index(taux_2_column)] = taux_2_value

        for i, header in enumerate(combined_headers):
            if 'Taux 2' in header:
                for row in combined_data:
                    if i < len(row): 
                        try:
                            taux_2_value = float(row[i])
                            row[i] = format(round(taux_2_value, 2), '.2f')
                        except (ValueError, TypeError): 
                            row[i] = ''




    def write_combined_csv(output_file, combined_headers, combined_data):
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(combined_headers)
            writer.writerows(combined_data)

    def extract_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    input_files = [os.path.join(processed_directory, filename) for filename in os.listdir(processed_directory) if filename.endswith('.csv')]

    input_files.sort(key=lambda f: extract_page_number(f))

    output_file = './combined_output.csv'

    combined_headers, combined_data = merge_csv_files(input_files)

    add_taux_2_columns(combined_headers, combined_data)

    write_combined_csv(output_file, combined_headers, combined_data)

    st.write("Combined CSV file with Taux 2 columns has been saved.")

    def update_headers(file_path, output_path):
        df = pd.read_csv(file_path, header=None)

        new_headers = []

        for column in df:
            column_values = df[column].astype(str)  
            if 'Abs.' in column_values.values:
                new_headers.append('Abs.')
            elif 'Date Equipe Hor.Abs.' in column_values.values:
                new_headers.append('Date Equipe Hor.Abs.')
            else:
                new_headers.append(df.columns[column]) 
        df.columns = new_headers

        df.to_csv(output_path, index=False)

    for file_path in os.listdir(rest_output_directory) :
        if filename.endswith('.csv'):
            file_path_upd= os.path.join(rest_output_directory,file_path)
            update_headers(file_path_upd,file_path_upd)


    def process_dataframe(df):
        absences_par_jour = 0
        absences_par_heure = 0.0

        columns_to_check = [col for col in df.columns if 'Abs.' in col or 'Date Equipe Hor.Abs.' in col]

        for col in columns_to_check:
            for cell in df[col].astype(str): 
                cell = cell.strip() 

                if 'AB' in cell:
                    absences_par_jour += 1 
                    match = re.search(r'(\d+(?:\.\d+)?)AB$', cell)
                    if match:
                        absences_par_heure += float(match.group(1))  

        return absences_par_jour, absences_par_heure

    def extract_page_number(filename):
        match = re.search(r'_page_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')  

    def generate_absences_report(input_directory, output_file):
        report_data = [
            ['Nom du Fichier', 'Absences par Jour', 'Absences par Heure']
        ]

        files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
        files.sort(key=lambda f: extract_page_number(f))

        for filename in files:
            file_path = os.path.join(input_directory, filename)
            absences_par_jour, absences_par_heure = process_csv_file(file_path)
            report_data.append([filename, absences_par_jour, absences_par_heure])

        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(report_data)

    absence_output_file = './Absences.csv'

    generate_absences_report(rest_output_directory, absence_output_file)

    

    matricules_file_path = "./CSV3/matricules/matricules.csv"
    combined_output_file_path = "./combined_output.csv"
    output_file_path = "./merged_output.csv"

    
    matricules = []
    with open(matricules_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for i, row in enumerate(reader):
            if i < 600 and row:  
                matricules.append(row[0])
            elif i >= 600 :
                break
    combined_data = []
    with open(combined_output_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        combined_data = [row for row in reader]

    if len(matricules) > len(combined_data) - 1:  
        st.write(len(matricules))
        st.write(len(combined_data)-1)
        raise ValueError("Le fichier de matricules contient plus de lignes que le fichier combined_output.")

    merged_data = []

    merged_data.append(["Matricule"] + combined_data[0])

    for i, row in enumerate(combined_data[1:], start=1):
        if i <= len(matricules):  
            matricule = matricules[i - 1]
            merged_data.append([matricule] + row)
        else:
            break

    with open(output_file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(merged_data)

    st.write(f"Tableau combiné avec les 3 premières matricules enregistré dans {output_file_path}.")

    main_table = pd.read_csv(main_table_path)
    cumul_data = pd.read_excel(uploaded_file_1, engine='openpyxl')
    info_salaries_data = pd.read_excel(uploaded_file_2, engine='openpyxl')
    absences_data = pd.read_csv(absences_path)

    def convert_to_float(value):
        if pd.notnull(value):  
            try:
                value_str = str(value).strip()

                if value_str.endswith('-'):
                    value_str = '-' + value_str[:-1]

                value_str = value_str.replace('.', '').replace(',', '.')

                return pd.to_numeric(value_str, errors='coerce')
            except ValueError:
                return value
        return value


    for col in main_table.columns:
        if main_table[col].dtype == 'object':
            main_table[col] = main_table[col].apply(convert_to_float)

    main_table['Matricule'] = main_table['Matricule'].astype(str)
    cumul_data['Matricule'] = cumul_data['Matricule'].astype(str)
    info_salaries_data['Matricule'] = info_salaries_data['Matricule'].astype(str)

    merged_df = main_table.merge(cumul_data, on='Matricule', how='left')
    final_df = merged_df.merge(info_salaries_data, on='Matricule', how='left')

    cols = list(final_df.columns)

    
    if 'Nom Prénom' in cols:
        cols.insert(1, cols.pop(cols.index('Nom Prénom')))
    final_df = final_df[cols]

    if 'Nom' in final_df.columns:
        final_df = final_df.drop(columns=['Nom'])

    if 'Nom du Fichier' in absences_data.columns:
        absences_data = absences_data.drop(columns=['Nom du Fichier'])

    if len(final_df) > len(absences_data):
        absences_data = absences_data.reindex(range(len(final_df))).fillna('')
    elif len(final_df) < len(absences_data):
        final_df = final_df.reindex(range(len(absences_data))).fillna('')

    concatenated_df = pd.concat([final_df, absences_data], axis=1)



    concatenated_df.to_csv(final_output_csv_path, index=False)

    csv_data = concatenated_df.to_csv(index=False).encode('utf-8')

    st.write(f"Fichier sauvegardé sous {final_output_csv_path}")

    st.download_button(
        label="Download fichier restructuré",
        data=csv_data,
        file_name="Fichier restrucuturé.csv",
        mime="text/csv"
    )
    csv_files_to_delete = glob.glob(os.path.join(output_dir, "**", "*.csv"), recursive=True)

    for csv_file in csv_files_to_delete:
        try:
            os.remove(csv_file)
        except Exception as e:
            st.write(f"Erreur lors de la suppression du fichier {csv_file}: {e}")