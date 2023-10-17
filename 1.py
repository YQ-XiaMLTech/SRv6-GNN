# import tensorflow as tf
#
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import random

for i in range(200):
    input_file_path = 'datasets/GEANT2/gravity_1/TM/TM-' + str(i)
    output_file_path = 'datasets/GEANT2/gravity_1/TM/TM-' + str(i)

    # Read the input file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Ensure we have more than 102 lines (2 header + 100 to delete)
    if len(lines) > 102:
        # Lines to keep (first two + 100 random from the rest)
        lines_to_keep = lines[:2] + random.sample(lines[2:], len(lines) - 102)
    else:
        print("File has insufficient lines to remove 100 from it.")
        lines_to_keep = lines  # Keep the file as is if not enough lines

    # Write the output file
    with open(output_file_path, 'w') as file:
        file.writelines(lines_to_keep)
