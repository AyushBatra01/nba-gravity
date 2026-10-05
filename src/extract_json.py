from src.unzip import unzip


def extract_json(path, output_dir="games"):
    """Extract a compressed game archive into the local games directory."""
    unzip(path, output_dir)
