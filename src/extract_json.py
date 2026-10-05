import py7zr

def extract_json(path):
    with py7zr.SevenZipFile(path, mode='r') as z:
        z.extractall(path="games")