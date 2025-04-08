stimuli_dict = {'Stimulus/S  1': 1,
'Stimulus/S  2': 2,
'Stimulus/S  3': 3,
'Stimulus/S  4': 4,
'Stimulus/S  5': 5,
'Stimulus/S  6': 6,
'Stimulus/S  8': 8,
'Stimulus/S  9': 9,
'Stimulus/S 64': 64,
'Stimulus/S 65': 65,
'Stimulus/S 66': 66,
'Stimulus/S 67': 67,
'Stimulus/S 68': 68,
'Stimulus/S 69': 69,
'Stimulus/S 70': 70,
'Stimulus/S 71': 71,
'Stimulus/S 72': 72,
'Stimulus/S 73': 73 }

response_mapping = {'1': 129, '65': 129,
                    '2': 130, '66': 130,
                    '3': 131, '67': 131,
                    '4': 132, '68': 132,
                    '5': 133, '69': 133,
                    '6': 134, '70': 134,
                    '8': 136, '72': 136,
                    '9': 137, '73': 137}

actual_mapping = {'New Segment/': 99999,
  'Stimulus/S  1': 1,
  'Stimulus/S  2': 2,
  'Stimulus/S  3': 3,
  'Stimulus/S  4': 4,
  'Stimulus/S  5': 5,
  'Stimulus/S  6': 6,
  'Stimulus/S  8': 8,
  'Stimulus/S  9': 9,
  'Stimulus/S 64': 64,
  'Stimulus/S 65': 65,
  'Stimulus/S 66': 66,
  'Stimulus/S 67': 67,
  'Stimulus/S 68': 68,
  'Stimulus/S 69': 69,
  'Stimulus/S 70': 70,
  'Stimulus/S 71': 71,
  'Stimulus/S 72': 72,
  'Stimulus/S 73': 73,
  'Stimulus/S129': 129,
  'Stimulus/S130': 130,
  'Stimulus/S131': 131,
  'Stimulus/S132': 132,
  'Stimulus/S133': 133,
  'Stimulus/S134': 134,
  'Stimulus/S136': 136,
  'Stimulus/S137': 137
                  }

matching_events = {'1': 65, '2': 66, '3': 67, '4': 68, '5': 69, '6': 70, '7': 71, '8': 72, '9': 73}


conditions = ['a1', 'a2', 'e1', 'e2']

sub_list = []
for i in range(1, 30, 1):
    # .zfill(2):
    # Adds leading zeros to the string until its length is 2 characters.
    string = f'sub{str(i).zfill(2)}'
    if string in ['sub06', 'sub07', 'sub09', 'sub12', 'sub16']:
        continue
    else:
        sub_list.append(string)

event_types = [
    'animal_sounds',
    'targets_with_valid_responses', 'targets_with_early_responses', 'targets_with_delayed_responses', 'targets_without_responses',
    'distractors_with_valid_responses', 'distractors_with_early_responses', 'distractors_with_delayed_responses', 'distractors_without_responses',
    'non_targets_target_with_valid_responses', 'non_targets_target_with_early_responses', 'non_targets_target_with_delayed_responses', 'non_targets_target_no_response',
    'non_targets_distractor_with_valid_responses', 'non_targets_distractor_with_early_responses', 'non_targets_distractor_with_delayed_responses', 'non_targets_distractor_no_response'
]

channels = ['motor', 'attention']

exceptions = ['sub06', 'sub16']

occipital_channels = ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "P5", "P6", "P7", "P8"]
motor_channels = ['C3', 'CP3', 'FC3', 'C4',  'CP4', 'Cz',  'FC4']