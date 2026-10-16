from pathlib import Path
from src.unzip import unzip

base_path = Path.cwd()
raw_game_path = base_path / "games-raw"
processed_game_path = base_path / "games"

for f in raw_game_path.iterdir():
    if f.is_file() and f.suffix == ".7z":
        unzip(f, processed_game_path)