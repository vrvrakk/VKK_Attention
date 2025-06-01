import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

# Base directory
plane = 'azimuth'
base_dir = rf"C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\trf\trf_testing\composite_model\single_sub\{plane}\all_stims"


####################################
# Predictor folders to include
predictors = ["envelopes", "onsets", "overlap_ratios", "events_proximity_pre", "events_proximity_post", "RTs"]

# Storage for mean r-values
mean_r_values = {"Predictor": [], "Target": [], "Distractor": []}
# Process each predictor folder
for predictor in predictors:
    predictor_path = os.path.join(base_dir, predictor)
    files = os.listdir(predictor_path)

    target_files = [f for f in files if f"subjectwise_crossval_rvals_{plane}_target_stream" in f and f.endswith(".npy")]
    distractor_files = [f for f in files if f"subjectwise_crossval_rvals_{plane}_distractor_stream" in f and f.endswith(".npy")]

    if not target_files or not distractor_files:
        print(f"Skipping {predictor} (missing files)")
        continue

    r_target_dict = np.load(os.path.join(predictor_path, target_files[0]), allow_pickle=True).item()
    r_distractor_dict = np.load(os.path.join(predictor_path, distractor_files[0]), allow_pickle=True).item()

    # Append mean r-values
    # Extract and average r-values
    mean_r_values["Predictor"].append(predictor)
    mean_r_values["Target"].append(np.mean(list(r_target_dict.values())))
    mean_r_values["Distractor"].append(np.mean(list(r_distractor_dict.values())))

# Plotting
x = np.arange(len(mean_r_values["Predictor"]))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, mean_r_values["Target"], width, label='Target Stream', alpha=0.8)
ax.bar(x + width/2, mean_r_values["Distractor"], width, label='Distractor Stream', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(mean_r_values["Predictor"], rotation=30)
ax.set_ylabel("Mean Correlation (r)")
ax.set_title(f"Mean TRF Model Performance per Predictor ({plane} Plane)")
ax.legend()
plt.tight_layout()
plt.show()

###################


# Composite model folders to include (excluding 'single_preds')
predictor_folders = [
    "on_en",
    'on_en_ov',
    'on_en_ev1',
    'on_en_ev2',
    'on_en_ov_ev1_RT',
    'on_en_ov_ev2_RT',
    "on_en_RT",
    "on_en_RT_ov",
    "on_en_RT_ov_ev1_ev2"
]

# === STORAGE ===
mean_r_values = {"Predictor": [], "Target": [], "Distractor": []}

# === PROCESSING LOOP ===
for predictor in predictor_folders:
    predictor_path = os.path.join(base_dir, predictor)
    files = os.listdir(predictor_path)
    # Identify files
    target_file = [f for f in files if f"subjectwise_crossval_rvals_{plane}_target_stream" in f and f.endswith(".npy")]
    distractor_file = [f for f in files if f"subjectwise_crossval_rvals_{plane}_distractor_stream" in f and f.endswith(".npy")]

    if not target_file or not distractor_file:
        print(f"Skipping {predictor} (missing files)")
        continue

    r_target = np.load(os.path.join(predictor_path, target_file[0]), allow_pickle=True).item()
    r_distractor = np.load(os.path.join(predictor_path, distractor_file[0]), allow_pickle=True).item()

    # Save means
    mean_r_values["Predictor"].append(predictor)
    mean_r_values["Target"].append(np.mean(list(r_target.values())))
    mean_r_values["Distractor"].append(np.mean(list(r_distractor.values())))

# === PLOT ===
x = np.arange(len(mean_r_values["Predictor"]))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, mean_r_values["Target"], width, label='Target', color='royalblue')
ax.bar(x + width/2, mean_r_values["Distractor"], width, label='Distractor', color='darkorange')

ax.set_xticks(x)
ax.set_xticklabels(mean_r_values["Predictor"], rotation=45, ha="right")
ax.set_ylabel('Mean Correlation (r)')
ax.set_title(f'TRF Composite Models ({plane}) - Mean r-values per Predictor Combo')
ax.legend()
plt.tight_layout()
plt.show()