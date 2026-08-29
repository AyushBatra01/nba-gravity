import py7zr

def unzip(input_path, output_dir):
    with py7zr.SevenZipFile(input_path, mode='r') as z:
        z.extractall(path=output_dir)