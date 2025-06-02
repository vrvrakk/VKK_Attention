import subprocess
from pathlib import Path

default_path = Path.cwd()


subprocess.run(["python", '-m', "TRF_predictors.onsets"], cwd=default_path)

subprocess.run(["python", "-m", "TRF_predictors.extract_envelope"], cwd=default_path)


subprocess.run(["python", "-m", "TRF_predictors.overlap_ratios"], cwd=default_path)
subprocess.run(["python", "-m", "TRF_predictors.events_proximity"], cwd=default_path)
subprocess.run(["python", "-m", "TRF_predictors.RTs"], cwd=default_path)
subprocess.run(["python", "-m", "TRF_predictors.mask_bad_segments"], cwd=default_path)

