import os
from datetime import datetime

# Function to extract release date from the header
def extract_release_date(header):
    lines = header.split('\n')
    for line in lines:
        if "release date:" in line.lower():
            date_str = line.split(":")[1].strip()
            # make sure there is no EBOOK number
            date_str = date_str.split('[')[0].strip()
            print(date_str)
            return datetime.strptime(date_str, "%B %d, %Y")
    print('failed')
    return None

# Directory containing your txt files
files_directory = "./"

# Desired date to filter files
desired_date = datetime(2023, 2, 28)

# List to store filtered file paths
filtered_files = []

# Loop through each file in the directory
for filename in os.listdir(files_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(files_directory, filename)
        with open(file_path, 'rb') as file:
            header = file.read(1000)  # Read the first part of the file as bytes
            try:
                header = header.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    header = header.decode('latin-1')
                except UnicodeDecodeError:
                    continue  # Skip this file if decoding fails
            try:
                release_date = extract_release_date(header)
                if release_date and release_date > desired_date:
                    filtered_files.append(file_path)
                    print(file_path)
            except Exception as e:
                print(file_path, e)

# Copy filtered files to a destination directory
destination_directory = "./filtered_28022023/"
for file_path in filtered_files:
    destination_path = os.path.join(destination_directory, os.path.basename(file_path))
    with open(file_path, 'rb') as src_file, open(destination_path, 'wb') as dest_file:
        dest_file.write(src_file.read())

