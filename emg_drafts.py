'''

           def compare_multiple_epoch_powers(epochs_baseline, combined_baseline, combined_distractor_no_response_epochs, combined_target_response_epochs, frequencies, n_cycles,
                                  label1="True Baseline", label2="Non-Target Stimuli", label3="Distractors", label4="Targets",
                                  tmin_early=0.0, tmax_early=0.4, tmin_late=0.4, tmax_late=0.9):
    """
    Compares the power trends of four different epoch types across alpha, beta, and gamma bands.

    Parameters:
        epochs_baseline (Epochs): True baseline epochs.
        combined_baseline (Epochs): Combined baseline epochs for non-target stimuli.
        combined_distractor_no_response_epochs (Epochs): Combined distractor epochs with no response.
        combined_target_response_epochs (Epochs): Combined target response epochs.
        frequencies (ndarray): Array of frequency values.
        n_cycles (ndarray): Array of cycles for Morlet wavelet analysis.
        tmin_early (float): Start time for the early phase window.
        tmax_early (float): End time for the early phase window.
        tmin_late (float): Start time for the late phase window.
        tmax_late (float): End time for the late phase window.
    """
    # Define frequency bands
    alpha_band = [1, 10]
    beta_band = [10, 20]
    gamma_band = [20, 30]


    # Compute TFR (time-frequency representation) using Morlet wavelets for each set of epochs
    power_baseline = mne.time_frequency.tfr_morlet(epochs_baseline, freqs=frequencies, n_cycles=n_cycles, return_itc=False)
    power_combined_baseline = mne.time_frequency.tfr_morlet(combined_baseline, freqs=frequencies, n_cycles=n_cycles, return_itc=False)
    power_distractors = mne.time_frequency.tfr_morlet(combined_distractor_no_response_epochs, freqs=frequencies, n_cycles=n_cycles, return_itc=False)
    power_targets = mne.time_frequency.tfr_morlet(combined_target_response_epochs, freqs=frequencies, n_cycles=n_cycles, return_itc=False)

    # Define early and late windows for all epochs
    def extract_windows(power):
        early_window = power.copy().crop(tmin=tmin_early, tmax=tmax_early)
        late_window = power.copy().crop(tmin=tmin_late, tmax=tmax_late)
        return early_window, late_window

    early_baseline, late_baseline = extract_windows(power_baseline)
    early_combined_baseline, late_combined_baseline = extract_windows(power_combined_baseline)
    early_distractors, late_distractors = extract_windows(power_distractors)
    early_targets, late_targets = extract_windows(power_targets)

    # Extract power for alpha, beta, and gamma bands
    def extract_band_power(window, band):
        return window.copy().crop(fmin=band[0], fmax=band[1]).data.mean(axis=2)

    alpha_frequencies = frequencies[(frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1])]
    beta_frequencies = frequencies[(frequencies >= beta_band[0]) & (frequencies <= beta_band[1])]
    gamma_frequencies = frequencies[(frequencies >= gamma_band[0]) & (frequencies <= gamma_band[1])]

    # Plot comparison of power trends for early and late phases in alpha, beta, and gamma bands
    plt.figure(figsize=(15, 8))

    def plot_power(early_power, late_power, color, label, linestyle="-"):
        plt.plot(alpha_frequencies, early_power[0].flatten(), label=f'Early {label}', color=color,linestyle=linestyle)
        plt.plot(alpha_frequencies, late_power[0].flatten(),label=f'Late {label}', color=color, linestyle="--")
        plt.plot(beta_frequencies, early_power[1].flatten(), color=color,
                 linestyle=linestyle)
        plt.plot(beta_frequencies, late_power[1].flatten(), color=color, linestyle="--")
        plt.plot(gamma_frequencies, early_power[2].flatten(), color=color,
                 linestyle=linestyle)
        plt.plot(gamma_frequencies, late_power[2].flatten(), color=color, linestyle="--")

    # Plot for all sets of epochs
    plot_power([extract_band_power(early_baseline, alpha_band), extract_band_power(early_baseline, beta_band),
                extract_band_power(early_baseline, gamma_band)],
               [extract_band_power(late_baseline, alpha_band), extract_band_power(late_baseline, beta_band),
                extract_band_power(late_baseline, gamma_band)], color="red", label=label1)

    plot_power([extract_band_power(early_combined_baseline, alpha_band), extract_band_power(early_combined_baseline, beta_band),
                extract_band_power(early_combined_baseline, gamma_band)],
               [extract_band_power(late_combined_baseline, alpha_band), extract_band_power(late_combined_baseline, beta_band),
                extract_band_power(late_combined_baseline, gamma_band)], color="blue", label=label2)

    plot_power([extract_band_power(early_distractors, alpha_band), extract_band_power(early_distractors, beta_band),
                extract_band_power(early_distractors, gamma_band)],
               [extract_band_power(late_distractors, alpha_band), extract_band_power(late_distractors, beta_band),
                extract_band_power(late_distractors, gamma_band)], color="green", label=label3)

    plot_power([extract_band_power(early_targets, alpha_band), extract_band_power(early_targets, beta_band),
                extract_band_power(early_targets, gamma_band)],
               [extract_band_power(late_targets, alpha_band), extract_band_power(late_targets, beta_band),
                extract_band_power(late_targets, gamma_band)], color="purple", label=label4)

    # Add vertical lines for band boundaries
    plt.axvline(alpha_band[1], color="black", linestyle="--", label="Alpha/Beta Boundary")
    plt.axvline(beta_band[1], color="black", linestyle="--", label="Beta/Gamma Boundary")

    # Labels and title
    plt.title(f"Power Trends Across Alpha, Beta, and Gamma Bands for All Epoch Types (Early vs Late Phases)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (log ratio)")
    plt.legend(loc="best")

    # Show plot
    plt.savefig(psd_path/'all_compared.png')



            # Step 1: Initialize cumulative counts for all blocks
    all_target_stimuli = 0
    all_distractor_stimuli = 0
    all_target_no_response = 0
    all_distractor_no_response = 0
    all_button_presses = 0
    all_correct_responses = 0
    all_distractor_responses = 0
    all_invalid_responses = 0
    all_target_temptations = 0
    all_distractor_temptations = 0
    all_errors = 0
    all_target_misses = 0
        # load the csv tables, related to the condition and index: they contain vmrk information
        vmrk_files = []
        for files in df_path.iterdir():
            if files.is_file and f'{condition}_{index}' in files.name:
                vmrk_files.append(files)
        vmrk_dfs = {}
        for idx, dfs in enumerate(vmrk_files):
            vmrk_df = pd.read_csv(dfs, delimiter=',')
            vmrk_dfs[dfs.name[:-4]] = vmrk_df
            # here we read each csv file: target resposnes, distractor responses, invalid responses
            # and target and distractor events without response
        # correct responses, distractor responses: 1
        # no responses: 0
        # invalid responses: 3
        # use baseline epochs as, well, baseline

        concatenated_vmrk_dfs = pd.concat(vmrk_dfs)
        vmrk_dfs_sorted = concatenated_vmrk_dfs.sort_values(by=['Unnamed: 0'], ascending=True)
        vmrk_dfs = vmrk_dfs_sorted.rename(columns={'Unnamed: 0': 'Index'})
        del concatenated_vmrk_dfs, vmrk_dfs_sorted


        # Create thresholds:
        # Calculate the baseline derivative:
        def baseline_normalization(emg_epochs, tmin, tmax):
            emg_epochs.load_data()  # This ensures data is available for manipulation
            emg_window = emg_epochs.copy().crop(tmin, tmax)  # select a time window within each epoch, where a response to stimulus is expected
            emg_data = emg_window.get_data(copy=True)  # Now it's a NumPy array
            emg_derivative = np.diff(emg_data, axis=-1)  # Derivative across time of baseline
            # The np.diff() function calculates the derivative of emg_data along the time axis (axis=-1).
            # This results in a new array, emg_derivative, with a shape of (n_epochs, n_channels, n_samples - 1).
            # Z-score normalization
            # get baseline z scores
            emg_derivative_z = (emg_derivative - np.mean(emg_derivative, axis=-1, keepdims=True)) / np.std(
                emg_derivative, axis=-1, keepdims=True)
            # reduce its dimension
            emg_reduced = np.squeeze(emg_derivative_z)
            # get z scores variance and rms
            emg_var = np.var(emg_reduced, axis=-1)
            emg_rms = np.sqrt(np.mean(np.square(emg_reduced), axis=-1))
            return emg_data, emg_var, emg_rms, emg_derivative, emg_reduced

        baseline_emg_data, baseline_var, baseline_rms, baseline_emg_derivative, baseline_emg_derivative_z = baseline_normalization(epochs_baseline, tmin=0.0, tmax=0.7)
        baseline_mean = np.squeeze(np.mean(baseline_emg_derivative, axis=-1, keepdims=True))
        baseline_std = np.squeeze(np.std(baseline_emg_derivative, axis=-1, keepdims=True))

        axis=0: Refers to the first dimension (rows or the outermost dimension).
           axis=1: Refers to the second dimension (columns or the second layer of dimensions).
           axis=-1: Refers to the last dimension (often time samples or individual values within a feature).

         Extract data from MNE Epochs object (shape will be [n_epochs, n_channels, n_samples])
        def z_normalization(emg_epochs, baseline_mean, baseline_std, tmin, tmax):
            emg_epochs.load_data()  # Ensure data is loaded for manipulation
            emg_window = emg_epochs.copy().crop(tmin, tmax)  # Crop epochs to desired window
            emg_data = emg_window.get_data(copy=True)  # Extract data as a NumPy array

            # Compute the derivative across the time axis
            emg_derivative = np.diff(emg_data, axis=-1)

            # get an average baseline mean and std:
            baseline_mean = np.mean(baseline_mean, axis=0)  # use global baseline mean
            baseline_std = np.mean(baseline_std, axis=0)  # Use global baseline std
            # Collapse over epochs axis to get a single mean and std
            # Apply normalization using the correct broadcasting for single channel data:
            #  NumPy automatically "expands" one or more of the arrays
            #  to make their shapes compatible without making additional copies of the data
            emg_derivative_z = (emg_derivative - baseline_mean) / baseline_std
            emg_reduced = np.squeeze(emg_derivative_z)
            # get z-scores variance, RMS and std
            emg_var = np.var(emg_reduced, axis=-1)
            emg_rms = np.sqrt(np.mean(np.square(emg_reduced), axis=-1))
            emg_std = np.std(emg_reduced, axis=-1)  # across all values

            event_times = emg_window.times
            timepoints = []
            for idx, row in enumerate(emg_window.events):
                epoch_times = int(emg_window.events[idx][0]) / 500
                absolute_times = epoch_times + event_times  # get actual times of epoch events
                timepoints.append(absolute_times)
            absolute_timepoints = [times[:-1] for times in timepoints]
            epochs_z_scores_dfs = {}
            # Iterate through each epoch and its corresponding time points
            for epoch_idx, (epoch_data, time_array) in enumerate(zip(emg_reduced, absolute_timepoints)):
                epoch_df = pd.DataFrame({
                    'Timepoints': time_array,  # Create a column for timepoints
                    'Z-score': epoch_data,  # Create a column for z-scores
                    'Variance': emg_var[epoch_idx],  # Add variance for the entire epoch (broadcasted)
                    'RMS': emg_rms[epoch_idx],  # Add RMS for the entire epoch (broadcasted)
                    'STD': emg_std[epoch_idx]  # Add STD for the entire epoch (broadcasted)
                })
                # Append the DataFrame to the list
                epochs_z_scores_dfs[epoch_idx] = epoch_df
            return epochs_z_scores_dfs, emg_reduced, emg_var, emg_rms, emg_std

        # for a window of 350 samples (0.7s)
        target_z_scores_dfs, target_emg_z, target_var, target_rms, target_std = z_normalization(target_epochs, baseline_mean, baseline_std, tmin=0.2, tmax=0.9)  # , target=target[0]
        distractor_z_score_dfs, distractor_emg_z, distractor_var, distractor_rms, distractor_std = z_normalization(distractor_epochs, baseline_mean, baseline_std, tmin=0.2, tmax=0.9) # target=target[1]
        response_z_scores_dfs, response_emg_z, response_var, response_rms, response_std = z_normalization(response_epochs, baseline_mean, baseline_std, tmin=-0.2, tmax=0.5) # target=target[2]

        # True Response Threshold:
        # You can use the z-transformed EMG derivative from response epochs
        # to establish a threshold for classifying true responses.
        # Typically, you might select a threshold that is 1 standard deviation above the baseline
        # (i.e., a z-score > 1.0).
        # No Response Threshold: The baseline epochs can help define a threshold
        # for identifying no responses (i.e., z-scores close to 0).
        # Partial Response: Anything in between could be considered a partial response,
        # where the z-scores are not high enough to be classified as a full true response,
        # but there is still some activity.
        # Upper threshold for true responses (from response epochs)#
        def adaptive_threshold(response_emg_z, baseline_emg_derivative_z, response_std):
            r_std = np.mean(response_std)
            # get fixed response threshold for participant:
            response_threshold = np.abs(np.max(response_emg_z)) / (r_std)
            # get global baseline mean and std:
            # fixed baseline threshold for participant and condition:
            b_std = np.std(baseline_emg_derivative_z)
            no_response_threshold = np.abs(np.min(baseline_emg_derivative_z)) / 2 * b_std
            return response_threshold, no_response_threshold


        def classify_emg_epochs(emg_z_scores_dfs, response_threshold, no_response_threshold):
            classifications_list = []
            for epochs_df, df in emg_z_scores_dfs.items():
                # Step 1: Calculate the absolute values of the z-scores
                z = np.abs(df['Z-score'].values)  # Calculate absolute values for the single row

                # Step 2: Use np.where to find the index of the max absolute z-score
                max_pos = np.where(z == np.max(z))  # Returns a tuple of (row indices, column indices)

                # Step 3: Since there is only one row, get the max column index
                max_column_index = max_pos[0][0]  # Extract the first column index from the tuple

                # Step 4: Use the column index to get the corresponding timepoint
                timepoint = df['Timepoints'][max_column_index]  # Retrieve timepoint using column index

                # Step 5: Extract the exact z-score value for printing (optional, since you have abs_vals)
                max_z_score = z[max_column_index]
                if max_z_score >= response_threshold:
                    classifications_list.append((epochs_df, "True Response", max_z_score, timepoint))
                elif max_z_score <= no_response_threshold:
                    classifications_list.append((epochs_df, "No Response", max_z_score, timepoint))
                else:
                    classifications_list.append((epochs_df, "Partial Response", max_z_score, timepoint))
            classifications = pd.DataFrame(classifications_list, columns=['Epoch', 'Response', 'Z-score', 'Timepoint'])

            return classifications

        # Classify target and distractor epochs separately
        response_threshold, no_response_threshold = adaptive_threshold(response_emg_z, baseline_emg_derivative_z, response_std)
        target_classifications = classify_emg_epochs(target_z_scores_dfs, response_threshold, no_response_threshold)
        distractor_classifications = classify_emg_epochs(distractor_z_score_dfs, response_threshold, no_response_threshold)


        def verify_and_refine_classification_all_conditions(classifications, vmrk_dfs, time_window=0.9, target=''):
            """
            Verifies classifications by checking if their timepoints match the corresponding
            vmrk_dfs responses within a given time window.

            Args:
            - classifications: DataFrame with initial classifications.
            - vmrk_dfs: DataFrame with event markers.
            - time_window: Time difference tolerance for matching (default is 0.9 seconds).
            - response: The type of response to verify in the classifications DataFrame.
            - vmrk_response: The corresponding vmrk_df response (e.g., 1 for True Response, 0 for No Response).

            Returns:
            - Updated classification DataFrame with a 'Match' column indicating matched rows.
            """

            # Create a copy of the initial classification DataFrame to modify
            refined_classification_df = classifications.copy()

            # Define mappings for classification types and corresponding `vmrk_response` values
            classification_conditions = {
                'True Response': [0, 1, 2],  # True Response can match both target (1) and distractor (2) responses
                'No Response': [0, 1, 2],
                'Partial Response': [0, 1, 2],  # Partial Responses can match with 0, 1, 2
            }

            # Define the match labels based on the `vmrk_response` value and the classification type
            match_labels = {
                # Solid Cases
                (1, 'True Response'): 'target button press',  # A correct button press to the target stimulus
                (0, 'No Response'): 'no response',  # No response to a presented stimulus (target or distractor)
                (2, 'True Response'): 'distractor button press',  # A button press response to a distractor stimulus

                # Partial Responses:
                (0, 'Partial Response'): 'response temptation',  # A partial response to a target stimulus
                (1, 'Partial Response'): 'weak target button press',  # A partial response to a distractor stimulus
                (2, 'Partial Response'): 'weak distractor button press',  # A partial response classified as invalid

                # Other:
                (0, 'True Response'): 'invalid response',
                (1, 'No Response'): 'weak target button press',
                (2, 'True Response'): 'weak distractor button press',  # A strong response but considered invalid
            }
            # Initialize the 'Match' column if not already present
            if 'Match' not in refined_classification_df.columns:
                refined_classification_df['Match'] = 'Unmatched'  # Default value for unmatched rows

            # Step 1: Iterate through each condition type and corresponding `vmrk_response` values
            for classification_type, vmrk_response_values in classification_conditions.items():
                # true, no and partial response, 0,1,2,3 in dict:
                # Extract rows from refined_classification_df matching the given classification type
                condition_responses_df = refined_classification_df[
                    refined_classification_df['Response'] == classification_type]
                # print(condition_responses_df)

                # Step 2: Iterate over the vmrk_response_values for the given classification_type
                # Apply proper parentheses to the logical condition
                for vmrk_response_value in vmrk_response_values:
                    # Apply proper parentheses to the logical condition -> rows of interest
                    vmrk_condition_responses_df = vmrk_dfs[
                        (vmrk_dfs['Response'] == vmrk_response_value) & (vmrk_dfs['Stimulus Type'].str.strip() == target)
                        ]
                    # Step 3: Match `condition_responses_df` in `refined_classification_df` with `vmrk_dfs` based on timepoints
                    match_results = []  # Store results of each verification for review

                    # Step 4: Iterate over each response in `condition_responses_df`
                    for idx, row in condition_responses_df.iterrows():
                        # Get the timepoint of the current Response in refined_classification_df
                        response_time = row['Timepoint']

                        # Check if there's a matching timepoint in vmrk_condition_responses_df within the allowed tolerance
                        matching_rows = vmrk_condition_responses_df[
                            (vmrk_condition_responses_df['Timepoints'] >= response_time - time_window) &
                            (vmrk_condition_responses_df['Timepoints'] <= response_time + time_window)
                            ]
                        # Step 5: If at least one match is found, update with the corresponding match label
                        if not matching_rows.empty:
                            match_label = match_labels.get((vmrk_response_value, classification_type), 'Matched')
                            match_results.append((idx, match_label))
                        else:
                            match_results.append((idx, 'Unmatched'))

                    # Step 6: Update the existing 'Match' column with the new match labels
                    for idx, match_status in match_results:
                        if refined_classification_df.at[idx, 'Match'] in ['Unmatched',
                                                                          'Not Matched']:  # Update only if not already matched
                            refined_classification_df.at[idx, 'Match'] = match_status

            return refined_classification_df


        target_refined_classifications = verify_and_refine_classification_all_conditions(target_classifications, vmrk_dfs, target=target_stream)
        distractor_refined_classifications = verify_and_refine_classification_all_conditions(distractor_classifications, vmrk_dfs, target=distractor_stream)


        def plot_emg_derivative_z(emg_derivative_z, target, epoch_idx=None):
            # Remove extra dimensions if necessary
            emg_derivative_z = np.squeeze(emg_derivative_z)  # This will reduce dimensions like (1, ...) to (...)

            # Create a figure
            plt.figure(figsize=(12, 6))

            # Plot each epoch individually without averaging
            for i in range(emg_derivative_z.shape[0]):
                plt.plot(emg_derivative_z[i].T, label=f'Epoch {i + 1}')

            # Set labels and title
            plt.title(f'EMG Derivative Z-Score (Individual Epochs) for {target}')
            plt.xlabel('Time (samples)')
            plt.ylabel('Z-Score')
            plt.legend(loc='upper right')
            # else:
            #     # If no epoch is specified, flatten and plot all epochs
            #     emg_data_flat = np.mean(emg_derivative_z, axis=1)  # Mean across channels
            #     plt.figure(figsize=(10, 6))
            #     for i in range(emg_data_flat.shape[0]):  # Iterate over epochs
            #         plt.plot(emg_data_flat[i], label=f'Epoch {i + 1}')
            #     plt.title('EMG Derivative Z-Score (All Epochs)')
            #     plt.xlabel('Time (samples)')
            #     plt.ylabel('Z-Score')
            #     plt.legend(loc='best')
            plt.savefig(z_figs / f'{sub_input}_{condition}_{target}_{index}_z_derivatives')

            #
            # # Call the function to plot all epochs


        plot_emg_derivative_z(target_emg_z, target=target[0])
        plot_emg_derivative_z(distractor_emg_z, target=target[1])
        plot_emg_derivative_z(response_emg_z, target=target[2])
        plot_emg_derivative_z(baseline_emg_derivative_z, target='baseline')


        def summarize_classifications(target_refined_classifications, distractor_refined_classifications, vmrk_dfs):
            # Step 1: Summarize the counts for each match label
            total_stim = vmrk_dfs[vmrk_dfs['Stimulus Type'] == target_stream]
            total_target_stim_count = len(total_stim)
            total_distractor_stim_count = len(vmrk_dfs[vmrk_dfs['Stimulus Type'] == distractor_stream])
            target_summary_counts = target_refined_classifications['Match'].value_counts()
            distractor_summary_counts = distractor_refined_classifications['Match'].value_counts()

            # Step 2: Define main categories based on match labels
            target_no_response = target_summary_counts.get('no response', 0)
            distractor_no_response = distractor_summary_counts.get('no response', 0)
            target_button_presses = target_summary_counts.get(f'target button press', 0) + target_summary_counts.get('weak target button press', 0)
            target_invalid_responses = target_summary_counts.get('invalid response', 0)
            target_response_temptation = target_summary_counts.get('response temptation', 0)
            distractor_response_temptation = distractor_summary_counts.get('response temptation')
            distractor_button_presses = distractor_summary_counts.get(f'distractor button press', 0) + distractor_summary_counts.get(f'weak distractor button press', 0)
            distractor_invalid_responses = distractor_summary_counts.get(f'invalid response', 0)
            total_button_presses = target_button_presses + target_invalid_responses + distractor_button_presses + distractor_invalid_responses


            # Step 3: Create categories and percentages
            summary_data = {
                'Target No-Responses': target_no_response,
                'Distractor No-Response': distractor_no_response,
                'Target Button-Presses': target_button_presses,
                'Distractor Button-Presses': distractor_button_presses,
                'Total Invalid-Responses (target & distractor)': target_invalid_responses + distractor_invalid_responses,
                'Target Response-Temptation': target_response_temptation,
                'Distractor Response-Temptation': distractor_response_temptation
            }

            # Calculate percentages for each category with error handling
            if target_button_presses is not None and total_target_stim_count not in [None, 0]:
                correct_responses = target_button_presses
            else:
                correct_responses = 0  # Set to zero if None or zero denominator

            if (target_invalid_responses is not None and distractor_invalid_responses is not None
                    and total_button_presses not in [None, 0]):
                invalid_responses = target_invalid_responses + distractor_invalid_responses
            else:
                invalid_responses = 0

            if distractor_button_presses is not None and total_button_presses not in [None, 0]:
                distractor_responses = distractor_button_presses
            else:
                distractor_responses = 0

            if target_response_temptation is not None and total_target_stim_count not in [None, 0]:
                target_temptations = target_response_temptation
            else:
                target_temptations = 0

            if distractor_response_temptation is not None and total_distractor_stim_count not in [None, 0]:
                distractor_temptations = distractor_response_temptation
            else:
                distractor_temptations = 0

            if (target_invalid_responses is not None and distractor_invalid_responses is not None
                    and distractor_button_presses is not None and total_button_presses not in [None, 0]):
                total_errors = target_invalid_responses + distractor_invalid_responses + distractor_button_presses
            else:
                total_errors = 0

            if total_target_stim_count not in [None, 0] and target_button_presses is not None:
                total_misses = total_target_stim_count - target_button_presses
            else:
                total_misses = 0

            summary_counts = {'Total Target Stimuli': total_target_stim_count, 'Total Distractor Stimuli': total_distractor_stim_count,'Total Button Presses': total_button_presses, 'Correct Responses': correct_responses, 'Distractor Responses': distractor_responses, 'Invalid Responses (all)': invalid_responses,
                                   'Target Response-Readiness': target_temptations, 'Distractor Response-Readiness': distractor_temptations, 'Total Errors (all responses)': total_errors,
                                   'Total Target Misses': total_misses, 'Total Target No-Responses': target_no_response, 'Distractor No-Responses': distractor_no_response}

            return summary_data, summary_counts

        # Step 4: Summarize target and distractor classifications


        summary_data, summary_counts = summarize_classifications(target_refined_classifications, distractor_refined_classifications, vmrk_dfs)

        # Update cumulative counts for the entire condition
        all_target_stimuli += summary_counts['Total Target Stimuli']
        all_distractor_stimuli += summary_counts['Total Distractor Stimuli']
        all_button_presses += summary_counts['Total Button Presses']
        all_correct_responses += summary_counts['Correct Responses']
        all_distractor_responses += summary_counts['Distractor Responses']
        all_invalid_responses += summary_counts['Invalid Responses (all)']
        all_target_temptations += summary_counts['Target Response-Readiness']
        all_distractor_temptations += summary_counts['Distractor Response-Readiness']
        all_errors += summary_counts['Total Errors (all responses)']
        all_target_misses += summary_counts['Total Target Misses']
        all_target_no_response += summary_counts['Total Target No-Responses']
        all_distractor_no_response += summary_counts['Distractor No-Responses']




# Calculate Correct Responses as a percentage of total target stimuli
if all_target_stimuli > 0:
    correct_responses = (all_correct_responses / all_target_stimuli) * 100
else:
    correct_responses = 0

# Calculate Distractor Responses as a percentage of all button presses
if all_button_presses > 0:
    distractor_responses = (all_distractor_responses / all_button_presses) * 100
else:
    distractor_responses = 0

# Calculate Target Temptations as a percentage of total target stimuli
if all_target_stimuli > 0:
    target_temptations = (all_target_temptations / all_target_stimuli) * 100
else:
    target_temptations = 0

# Calculate Distractor Temptations as a percentage of total distractor stimuli
if all_distractor_stimuli > 0:
    distractor_temptations = (all_distractor_temptations / all_distractor_stimuli) * 100
else:
    distractor_temptations = 0

# Calculate Invalid Responses as a percentage of total stimuli (target + distractor)
total_stimuli = all_target_stimuli + all_distractor_stimuli
if total_stimuli > 0:
    invalid_responses = (all_invalid_responses / total_stimuli) * 100
else:
    invalid_responses = 0

# Calculate Total Errors as a percentage of target stimuli
if all_target_stimuli > 0:
    errors = (all_errors / all_target_stimuli) * 100
else:
    errors = 0
if all_target_no_response > 0:
    target_no_responses = (all_target_no_response / all_target_stimuli) * 100
if all_distractor_no_response > 0:
    distractor_no_responses = (all_distractor_no_response / all_distractor_stimuli) * 100
# Calculate Misses as a percentage of target stimuli
if all_target_stimuli > 0:
    misses = ((all_target_stimuli - all_correct_responses) * 100) / all_target_stimuli
else:
    misses = 0

# Step 3: Create a summary dictionary for plotting
final_summary_percentages = {
    'Target Responses': correct_responses,
    'Distractor Responses': distractor_responses,
    'Target No-Responses': target_no_responses,
    'Distractor No-Responses': distractor_no_responses,
    #'Invalid Responses (all)': invalid_responses,
    'Target Response-Readiness': target_temptations,
    'Distractor Response-Readiness': distractor_temptations,
    'Total Errors (all false responses)': errors }
    #'Total Target Misses': misses

def plot_combined_summary_percentages(final_summary_percentages):
    """
    Plots a bar chart to visualize the combined percentages across all blocks.

    Args:
    - summary_percentages: Dictionary containing calculated percentages for each classification type.
    """
    # Step 4.1: Extract categories and their corresponding percentages
    categories = list(final_summary_percentages.keys())
    percentages = list(final_summary_percentages.values())
    rainbow_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#8B00FF']
    # Step 4.2: Create a bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(categories, percentages, color=rainbow_colors)

    # Step 4.3: Add labels and values to each bar
    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')

    # Step 4.4: Customize the plot
    plt.xlabel('Response Type')
    plt.ylabel('Percentage (%)')
    plt.title('Summary of EMG Z-score Classifications Across All Blocks')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(percentages) + 5)  # Adjust y-axis limit for better label placement

    # Step 4.5: Show the plot
    plt.tight_layout()
    plt.savefig(class_figs / f'EMG_{condition}_all_classifications_new.png')

# Plot the final summary percentages
plot_combined_summary_percentages(final_summary_percentages)

plt.close('all') # so python shuts the hell up


 # def plot_summary_percentages(summary_percentages):
        #     """
        #     Plots a bar chart to visualize the summarized classification percentages.
        #
        #     Args:
        #     - summary_percentages: Dictionary containing calculated percentages for each classification type.
        #     """
        #     # Step 1: Extract categories and their corresponding percentages
        #     categories = list(summary_percentages.keys())
        #     percentages = list(summary_percentages.values())
        #
        #     # Step 2: Create a bar plot
        #     plt.ioff()
        #     plt.figure(figsize=(14, 8))
        #     bars = plt.bar(categories, percentages, color=['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#8c8c8c'])
        #
        #     # Step 3: Add labels and values to each bar
        #     for bar, percentage in zip(bars, percentages):
        #         yval = bar.get_height()
        #         plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')
        #
        #     # Step 4: Customize the plot
        #     plt.xlabel('Response Type')
        #     plt.ylabel('Percentage (%)')
        #     plt.title('Summary of EMG Z-score Classifications')
        #     plt.xticks(rotation=45, ha='right')
        #     plt.ylim(0, max(percentages) + 5)  # Adjust y-axis limit for better label placement
        #
        #     # Step 5: Show the plot
        #     plt.tight_layout()
        #     plt.savefig(class_figs / f'{condition}_{index}_EMG_classifications.png')

        # Run the plotting function using your calculated summary percentages
        # plot_summary_percentages(summary_percentages)



                    def count_true_responses(df):
                """Helper function to count the number of True Responses in a DataFrame."""
                return len(df[df['Response'] == 'True Response'])

            initial_true_responses = count_true_responses(refined_classification_df)
            # Get the true response count from the VMRK DataFrame (markers)
            marker_true_responses = len((vmrk_dfs[(vmrk_dfs['Stimulus Type'] == target_stream) & (vmrk_dfs['Response'] == 1)]))
            print(f"Initial True Responses: {initial_true_responses}, Marker True Responses: {marker_true_responses}")
            # Iteratively refine the classification by adjusting the z-score threshold
            iterations = 0
            response_threshold = 10  # Start with an initial response threshold (adjust as needed)
            no_response_threshold = 2
            while initial_true_responses != marker_true_responses and iterations < 20:
                iterations += 1
                print(f"\nIteration: {iterations}")

                # Adjust thresholds based on mismatch (simple example: adjust response threshold down if too few True Responses)
                if initial_true_responses < marker_true_responses:
                    response_threshold *= 0.95  # Decrease threshold to get more True Responses
                    no_response_threshold *= 1.05  # Increase threshold to reduce False Positives
                else:
                    response_threshold *= 1.05  # Increase threshold to be stricter
                    no_response_threshold *= 0.95  # Decrease threshold for more No Responses

                print(
                    f"New Response Threshold: {response_threshold}, New No-Response Threshold: {no_response_threshold}")
                # Re-classify using the new thresholds
                refined_classification_df['Response'] = refined_classification_df['Z-score'].apply(
                    lambda z: "True Response" if z >= response_threshold else
                    "No Response" if z <= no_response_threshold else
                    "Partial Response"
                )
                # Re-count the number of True Responses after classification
                initial_true_responses = count_true_responses(refined_classification_df)
                print(f"Updated True Responses: {initial_true_responses}")
                # Final comparison to see if matching
            if initial_true_responses == marker_true_responses:
                print("\nRefinement successful! The number of True Responses matches the markers.")
            else:
                print("\nReached maximum iterations. The number of True Responses still does not match the markers.")
'''