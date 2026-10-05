import py7zr


def unzip(input_path, output_dir):
    """Extract a .7z archive into the requested output directory."""
    with py7zr.SevenZipFile(input_path, mode='r') as z:
        z.extractall(path=output_dir)

def extract_json(path, output_dir="games"):
    """Extract a compressed game archive into the local games directory."""
    unzip(path, output_dir)
