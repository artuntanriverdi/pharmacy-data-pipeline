import pandas as pd
import re
import numpy as np
import os

# Define input path
#input_path='C:\\Users\\gremotti\\Gmr S.r.L\\GMR-Progetti - Documenti\\2211-11 ClonazioneFarmacie\\PerPython\\'
input_path='/Users/artun/Desktop'
temp_path='C:\\GMR\\Progetti\\2211-11 ClonazioneFarmacie\\dati\DatiDM\\'


# Define the CSV file for anagrafica
anag_file = input_path+'stat_universo_202306280757.csv'
anag_file = temp_path+'stat_universo_202306280757.csv'

# Define the CSV file for MMAS
mmas_file = input_path+'PDF - MMAS Farmacie RP2202_open.csv'

# Define the CSV file for Comuni
comuni_file = input_path+'comuni.csv'

# Define xlsx file for catena
catena_path = input_path+'Info per MD.xlsx'

# Define the ssn file path
ssn_path =  input_path+'ssn_senza_prodotti_9__20230512.xlsx'

# directory to save intermediate results
directory = input_path+'savedtable\\'

# Function to strip non-numeric characters and convert to int
def strip_non_numeric_and_convert_to_int(s):
    stripped_value = re.sub(r'\D', '', str(s))  # Ensure input is string and remove non-digit characters
    return int(stripped_value) if stripped_value.isdigit() else None


# Function to assign 'phy_tipo' based on conditions
def assign_phy_tipo(row):
    minsal_phy_typ_dscr = row['minsal_phy_typ_dscr']
    minsal_phy_typ_code = str(
        row['minsal_phy_typ_code']).strip()  # Convert to string and strip leading and trailing spaces
    phy_id = row['minsal_phy_id']

    if minsal_phy_typ_code == '1':
        return 'O'
    elif minsal_phy_typ_dscr in ['Ordinaria', '-']:
        return 'O'
    elif minsal_phy_typ_dscr in ['Dispensario', 'Dispensario Stagionale', 'Dispensario stagionale']:
        return 'D'
    elif minsal_phy_typ_dscr == 'Succursale':
        return 'S'
    else:
        return 'N'


# Function to adjust month values for year 2023 and define new values
def adjust_months_and_define_values(df):
    df.loc[df['year'] == 2023, 'month'] += 12

    df.loc[df['month'] > 24, 'new_value'] = 'High'
    df.loc[(df['month'] >= 13) & (df['month'] <= 24), 'new_value'] = 'Medium'
    df.loc[df['month'] <= 12, 'new_value'] = 'Low'

    return df


try:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(anag_file, delimiter='|', low_memory=False)

    # Process 'cod_pdf' column to strip non-numeric characters and convert to int
    df['cod_pdf'] = df['cod_pdf'].apply(strip_non_numeric_and_convert_to_int)

    # Check missing value
    #missing_values = df['cod_pdf'].isnull().sum()
    #print(f"Number of missing values in 'cod_pdf' column: {missing_values}")


    # Convert 'minsal_phy_id' column to numeric
    df['minsal_phy_id'] = pd.to_numeric(df['minsal_phy_id'], errors='coerce')
    #missing_phy_id = df['minsal_phy_id'].isnull().sum()
    #print(f"Number of missing values in 'minsal_phy_id' column after conversion: {missing_phy_id}")

    # Process 'minsal_prod_perd' column to extract year and month
    df[['year', 'month']] = df['minsal_prod_perd'].str.split('-', expand=True).astype(int)

    # Assign 'phy_tipo' using the custom function
    df['phy_tipo'] = df.apply(assign_phy_tipo, axis=1)

    # Adjust month values for year 2023 and define new values
    df = adjust_months_and_define_values(df)

    # Set phy_tipo='O' if strip(minsal_phy_typ_code)='1' and phy_id is not missing
    df.loc[(df['phy_tipo'] == 'O') & (df['minsal_phy_id'].isna()), 'phy_tipo'] = ''

    # Filter the DataFrame based on the conditions specified
    anagrafica2 = df[df['phy_tipo'] == 'O']

    # Sort by 'phy_id', 'phy_tipo', 'month', and 'cod_pdf'
    anagrafica2 = anagrafica2.sort_values(by=['minsal_phy_id', 'phy_tipo', 'month', 'cod_pdf'])

    # Print the sorted and reordered DataFrame
    #print(anagrafica2)


except Exception as e:
    print(f"Error occurred: {e}")


# Define a function to convert columns using the best numeric format
def to_numeric_best(series):
    return pd.to_numeric(series, errors='coerce')


# Sort the data by minsal_phy_id
anagrafica2 = anagrafica2.sort_values(by='minsal_phy_id', kind='mergesort')

# Initialize first_mese and an empty list to store DataFrames for anagrafica3
first_mese = None
anagrafica3_list = []

# Iterate over the grouped data by minsal_phy_id
for minsal_phy_id, group in anagrafica2.groupby('minsal_phy_id'):
    group = group.copy()

    # Set first_mese
    group['first_mese'] = group['month'].iloc[0]

    # Convert columns to numeric
    group['CODPRO'] = to_numeric_best(group['minsal_phy_loc_prov_istat_code'])
    group['CODREG'] = to_numeric_best(group['minsal_phy_loc_reg_code'])
    group['nfatturato'] = to_numeric_best(group['fatturato'])
    group['geocluster_min'] = to_numeric_best(group['geocluster_ministeriale'])
    group['geocluster'] = to_numeric_best(group['geocluster_ufficiale'])
    group['istat_code'] = to_numeric_best(group['minsal_phy_loc_town_istat_code'])
    group['lat'] = to_numeric_best(group['minsal_phy_loc_lat'])
    group['long'] = to_numeric_best(group['minsal_phy_loc_lon'])

    # Append the last row of the group to anagrafica3_list
    anagrafica3_list.append(group.iloc[-1])

# Concatenate all the DataFrames in anagrafica3_list
anagrafica3 = pd.concat(anagrafica3_list, axis=1).T

# Rename columns
anagrafica3 = anagrafica3.rename(columns={
    'minsal_phy_loc_prov_name': 'PROVINCIA',
    'minsal_phy_loc_reg_name': 'REGIONE',
    'minsal_phy_loc_town_name': 'COMUNE'
})

# Keep only the desired columns
anagrafica3 = anagrafica3[[
    'CODPRO', 'CODREG', 'geocluster_min', 'geocluster', 'istat_code',
    'lat', 'long', 'nfatturato', 'month', 'PROVINCIA', 'REGIONE', 'COMUNE',
    'first_mese', 'minsal_phy_id', 'minsal_phy_dscr', 'minsal_phy_loc_adr', 'prov'
]]

# Read catena information


# Load the Excel file into a pandas DataFrame
df = pd.read_excel(catena_path, sheet_name='Input')

# Filter rows where 'Flag' is not empty
df = df[df['Flag'].notnull()]

# Convert 'Codice finale' to 'phy_id' (assuming 'Codice finale' is a column name in your Excel sheet)
df['phy_id'] = pd.to_numeric(df['Codice finale'], errors='coerce')

# Select and rename columns
df = df[['phy_id', 'Flag', 'Nome Catena']].rename(columns={'Flag': 'Catena', 'phy_id': 'minsal_phy_id'})

# Drop duplicates based on 'minsal_phy_id'
df.drop_duplicates(subset=['minsal_phy_id'], keep='first', inplace=True)

# Assign the resulting DataFrame to a new variable named 'Catena'
Catena = df

# Define the global variable
#MMAS = None

# Try different encodings
try:
   MMAS = pd.read_csv(mmas_file, sep=';', encoding='ISO-8859-1')

except UnicodeDecodeError as e:
    print(f"Error reading the file: {e}")
    # You can try another encoding if necessary
    # MMAS = pd.read_csv(input_file, sep=';', encoding='cp1252')
MMAS.rename(columns={'MinSal': 'minsal_phy_id'}, inplace=True)
MMAS.rename(columns={'Minsal': 'minsal_flag'}, inplace=True)

# Rename columns in MMAS to ensure consistency (if necessary)
MMAS = MMAS.rename(columns={'minsal_flag': 'minsal', 'Fatturato': 'mmas_fatt'}, errors='ignore')
MMAS = MMAS.drop(columns=['comune', 'provincia','pv_id'], errors='ignore')

# Ensure 'minsal_phy_id' is in all DataFrames
if 'minsal_phy_id' not in anagrafica3.columns:
    print("Error: 'minsal_phy_id' column not found in anagrafica3.")
elif 'minsal_phy_id' not in MMAS.columns:
    print("Error: 'minsal_phy_id' column not found in MMAS.")
elif 'minsal_phy_id' not in Catena.columns:
    print("Error: 'minsal_phy_id' column not found in Catena.")
else:
    # Merge MMAS with anagrafica3
    merged_MMAS_anagrafica3 = MMAS.merge(anagrafica3, on='minsal_phy_id', how='right')

    # Merge the result with Catena
    full_match = merged_MMAS_anagrafica3.merge(Catena, on='minsal_phy_id', how='left')
    full_match = full_match.drop_duplicates(subset='minsal_phy_id')

    # Optionally, sort by 'minsal_phy_id' and reset index (if desired)
    full_match_sorted = full_match.sort_values(by='minsal_phy_id').reset_index(drop=True)
    full_match_sorted.index = full_match_sorted.index + 1


# Replace empty strings in specific columns with 'NONE'
full_match['Catena'] = full_match['Catena'].replace('', 'NONE')
full_match['Nome Catena'] = full_match['Nome Catena'].replace('', 'NONE')

# If there are NaN values that you also want to replace with 'NONE', use fillna()
full_match['Catena'] = full_match['Catena'].fillna('NONE')
full_match['Nome Catena'] = full_match['Nome Catena'].fillna('NONE')


try:
    # Read the comuni files
    comuni = pd.read_csv(comuni_file, delimiter='|',encoding='latin1')

except Exception as e:
    print(f"Error occurred: {e}")


# Merge the tables on the 'minsal_phy_id' column
# ERROR: should use istat_cod and not minsal_phy_id
#full_match2 = pd.merge(full_match, comuni, on='minsal_phy_id', how='left')
full_match2 = pd.merge(full_match, comuni, on='istat_code', how='left')

# Read the ssn file
df = pd.read_excel(ssn_path)

# Ensure Minsal_id is treated as an integer
df['Minsal_id'] = df['Minsal_id'].astype(int)


# Function to split rows into pairs and add year and wave
def split_row(row, start_index):
    minsal_id = row['Minsal_id']
    values = row.drop('Minsal_id').tolist()
    new_rows = []
    original_ssn_qta_headers = []

    # Split values into pairs
    for i in range(0, len(values), 2):
        new_row = [minsal_id]
        new_row.extend(values[i:i + 2])
        new_rows.append(new_row)

        # Add original_ssn_qta_header for each new row
        header_index = start_index + (i // 2) * 2
        if header_index < len(df.columns):
            original_ssn_qta_headers.append(df.columns[header_index])
        else:
            original_ssn_qta_headers.append('')

    return new_rows, original_ssn_qta_headers


# Create a new list to store the split rows and headers
new_data = []
original_headers = []


# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    new_rows, headers = split_row(row, 1)  # Start index for ssn_qta headers is 1
    new_data.extend(new_rows)
    original_headers.extend(headers)

# Determine the maximum number of columns in the new DataFrame
max_columns = max(len(row) for row in new_data)

# Create new column names
new_columns = ['cod_pdf']
for i in range(1, max_columns):
    if i == 1:
        new_columns.append('ssn_qta')
    elif i == 2:
        new_columns.append('ssn')
    else:
        new_columns.append(f'value_{i}')

# Create a new DataFrame with the split rows
new_df = pd.DataFrame(new_data, columns=new_columns)

# Replace NaN values with empty strings
new_df = new_df.replace(np.nan, '', regex=True)

# Add the original_ssn_qta_header column
new_df['original_ssn_qta_header'] = original_headers

# Add new columns for year and month
new_df['year'] = new_df['original_ssn_qta_header'].str[4:8]  # Extract characters 5 to 8 (indexing starts at 0)
new_df['month'] = new_df['original_ssn_qta_header'].str[-2:].astype(int)

# Modify the month values if the year is 2023
new_df.loc[new_df['year'] == '2023', 'month'] += 12

# Ensure cod_pdf is treated as an integer
new_df['cod_pdf'] = new_df['cod_pdf'].astype(int)

# Reorder columns
new_df = new_df.reindex(columns=['cod_pdf', 'year', 'month', 'ssn_qta', 'ssn'])

# Save as a global variable
globals()['ssn1'] = new_df

ssn1 = ssn1.rename(columns={
    'cod_pdf': 'minsal_phy_id'
})


# Convert 'ssn_qta' to numeric, forcing errors to NaN
ssn1['ssn_qta'] = pd.to_numeric(ssn1['ssn_qta'], errors='coerce')

# Filter the ssn1 DataFrame based on phy_id in new_model7
ssn1_filtered = ssn1[ssn1['minsal_phy_id'].isin(full_match2['minsal_phy_id'])]

# Further filter rows where month is between 3 and 14
ssn1_filtered = ssn1_filtered[(ssn1_filtered['month'] >= 3) & (ssn1_filtered['month'] <= 14)]

# Reset index to ensure 'minsal_phy_id' is a column and not an index level
ssn1_filtered = ssn1_filtered.reset_index(drop=True)

# Group by 'minsal_phy_id' and perform the required calculations
def calculate_metrics(group):
    median_ssn_qta = group['ssn_qta'].median()
    group['ssn_qta_delta'] = group['ssn_qta'] / median_ssn_qta
    group['nobs_ssn'] = (group['ssn_qta'] > 0).sum()
    group['lug_ago_miss'] = ((group['month'].isin([7, 8])) & (group['ssn_qta'].isna())).sum()
    total_ssn_qta = group['ssn_qta'].sum()
    group['ssn_qta_index'] = 12 * group['ssn_qta'] / total_ssn_qta
    return group

# Perform groupby and apply calculation
ssn2 = ssn1_filtered.groupby('minsal_phy_id').apply(calculate_metrics)

# Ensure 'minsal_phy_id' is a column for sorting
ssn2 = ssn2.reset_index(drop=True)

# Sort by 'minsal_phy_id' and 'month'
ssn2 = ssn2.sort_values(by=['minsal_phy_id', 'month'])

# Reset index again after sorting
ssn2.reset_index(drop=True, inplace=True)

# Copy the DataFrame to create ssn3
ssn3 = ssn2.copy()

# Create ssn_qta_base column
ssn3['ssn_qta_base'] = ssn3['ssn_qta']

# Calculate nobs_ssn_tot
ssn3['nobs_ssn_tot'] = ssn3['nobs_ssn'] + ssn3['lug_ago_miss']

# Apply the conditions to update ssn_qta, nobs_ssn, and ssn_qta_index
def update_values(row):
    if not pd.isna(row['ssn_qta_delta']):
        if row['ssn_qta_delta'] > 2 or row['ssn_qta_delta'] < 0.3:
            row['ssn_qta'] = np.nan
            row['nobs_ssn'] = max(row['nobs_ssn'] - 1, 0)
            row['ssn_qta_index'] = np.nan
    if pd.isna(row['ssn_qta']) and row['month'] not in [7, 8]:
        row['ssn_qta_index'] = np.nan
    return row

# Apply the function to each row
ssn3 = ssn3.apply(update_values, axis=1)

# Filter rows where nobs_ssn_tot is equal to 12
filtered_df = ssn3[ssn3['nobs_ssn_tot'] == 12]

# Group by pdf_phy_id and month, and calculate the median of ssn_qta_index for each group
grouped_df = filtered_df.groupby(['minsal_phy_id', 'month'])['ssn_qta_index'].median().reset_index()

# Group by month to get the median of ssn_qta_index across all pdf_phy_id
common_sample_ssn = grouped_df.groupby('month')['ssn_qta_index'].median().reset_index()
common_sample_ssn.columns = ['month', 'median_ssn_qta_index']

common_sample_ssn = common_sample_ssn.rename(columns={
    'median_ssn_qta_index': 'ssn_qta_index'
})


# Assuming common_sample_ssn is your existing DataFrame
# Step 1: Calculate the total sum of all ssn_qta_index values
total_sum_ssn_qta_index = common_sample_ssn['ssn_qta_index'].sum()

# Step 2: Calculate the new common_sample_ssn_2 value
common_sample_ssn['common_sample_ssn_2'] =  12 * common_sample_ssn['ssn_qta_index'] / total_sum_ssn_qta_index

# Step 3: Create the new DataFrame with month and common_sample_ssn_2 values
common_sample_ssn2 = common_sample_ssn[['month', 'common_sample_ssn_2']]

# Drop the 'ssn_qta_index' column from ssn3
ssn3_dropped = ssn3.drop(columns=['ssn_qta_index'])

# Perform the left join on 'mese'
ssn4 = pd.merge(ssn3_dropped, common_sample_ssn2, on='month', how='left')

# Order by 'phy_id' and 'mese'
ssn4 = ssn4.sort_values(by=['minsal_phy_id', 'month'])
# Create a copy of ssn4 to avoid modifying the original DataFrame
ssn5 = ssn4.copy()

# Define the update function for 'ssn_qta_index'
def update_ssn_qta_index(row):
    if pd.isna(row['ssn_qta']):
        if row['month'] not in [7, 8]:
            return np.nan
        elif row['nobs_ssn'] <= 8:
            return np.nan
    return row['common_sample_ssn_2']

# Apply the function to update 'ssn_qta_index'
ssn5['common_sample_ssn_2'] = ssn5.apply(update_ssn_qta_index, axis=1)

# Group by 'phy_id' and calculate the required aggregations
ssn6 = ssn5.groupby('minsal_phy_id').agg({
    'ssn_qta': 'sum',
    'common_sample_ssn_2': 'sum',
    'ssn_qta_base': 'sum',
    'nobs_ssn': 'max'
}).reset_index()

# Rename columns to match the SAS output format
ssn6.columns = ['minsal_phy_id', 'sum_ssn_qta', 'sum_common_sample_ssn_2', 'sum_ssn_qta_base', 'max_nobs_ssn']

# Create a new DataFrame ssn7 based on ssn6
ssn7 = ssn6.copy()

# Add a new column ssn_qta_orig which is a copy of ssn_qta
ssn7['ssn_qta_orig'] = ssn7['sum_ssn_qta']

# Update the ssn_qta column using the specified formula
ssn7['sum_ssn_qta'] = 12 * ssn7['sum_ssn_qta'] / ssn7['sum_ssn_qta_base']

# Drop the _type_ and _freq_ columns if they exist
# Since these columns were not created in previous steps, this line is just for completeness
if '_type_' in ssn7.columns:
    ssn7 = ssn7.drop(columns=['_type_'])
if '_freq_' in ssn7.columns:
    ssn7 = ssn7.drop(columns=['_freq_'])

# Rename columns to match the original intent
ssn7.rename(columns={
    'ssn_qta_orig': 'ssn_qta',
    'sum_common_sample_ssn_2': 'ssn_qta_index',


    'sum_ssn_qta_index': 'ssn_qta_index',
    'sum_ssn_qta_base': 'ssn_qta_base',
    'max_nobs_ssn': 'nobs_ssn'
}, inplace=True)

ssn7 = ssn7.drop(columns=['sum_ssn_qta'])

new_order = ['minsal_phy_id', 'ssn_qta', 'ssn_qta_index', 'ssn_qta_base', 'nobs_ssn']
ssn7 = ssn7[new_order]

# Merge DataFrames on 'phy_id'
full_match3 = pd.merge(full_match2, ssn7, on='minsal_phy_id', how='inner')

# Merge DataFrames on 'minsal_phy_id'
full_match3 = pd.merge(full_match2, ssn7, on='minsal_phy_id', how='left')

# Remove duplicates based on 'minsal_phy_id'
full_match3 = full_match3.drop_duplicates(subset='minsal_phy_id')

file_name = 'full_match3.pkl'
file_path = os.path.join(directory, file_name)

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Save the DataFrame to a pickle file
full_match3.to_pickle(file_path)
