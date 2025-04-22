import subprocess
from pathlib import Path

default_path = Path.cwd()


subprocess.run(["python", str(default_path / "TRF_predictors/onsets.py")])
subprocess.run(["python", str(default_path / "TRF_predictors/extract_envelope.py")])
subprocess.run(["python", str(default_path / "TRF_predictors/overlap_ratios.py")])
subprocess.run(["python", str(default_path / "TRF_predictors/events_proximity.py")])
subprocess.run(["python", str(default_path / "TRF_predictors/RTs.py")])
subprocess.run(["python", str(default_path / "TRF_predictors/mask_bad_segments.py")])
