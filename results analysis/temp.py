# %%
import numpy as np
import os

# Set the target directory


# # Create an empty list to store the data from each file
# all_data = []

max_file_count = 20

# file_count = 0
# # Iterate over the files in the target directory
# for file_count, filename in enumerate(os.listdir(target_dir)):
#     if file_count == max_file_count:
#         break

#     # Construct the full file path
#     file_path = os.path.join(target_dir, filename)
#     print(file_path)

#     # Check if the file is a regular file (not a directory)
#     if os.path.isfile(file_path):
#         # Open the file and read the data
#         with open(file_path, "r") as f:
#             # Create an empty list to store the data from the current file
#             file_data = []
#             for line in f:
#                 # Split the line into columns
#                 columns = line.strip().split()

#                 # Convert the columns to integers and append them to the file data
#                 file_data.append(np.array([int(c) for c in columns], dtype=int))

#         # Append the data from the current file to the overall data list
#         all_data.append(file_data)

# # Concatenate the data from all the files into a single ndarray
# data = np.concatenate(all_data)

# %%

def load_experiment_data(target_dir = "results/EMA vs monotone"):
    # Create an empty list to store the data from each file
    all_data = []

    # Iterate over the files in the target directory
    for file_count, filename in enumerate(os.listdir(target_dir)):
        print(file_count)
        if file_count == max_file_count:
            break
        

        # Construct the full file path
        file_path = os.path.join(target_dir, filename)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Open the file and read the data
            with open(file_path, "r") as f:
                # Create an empty list to store the data from the current file
                file_data = []
                for line in f:
                    # Split the line into columns
                    columns = line.strip().split()

                    # Convert the columns to integers and append them to the file data
                    file_data.append(np.array([int(c) for c in columns], dtype=int))

            # Convert the data from the current file to an ndarray
            file_data = np.array(file_data)

            # Append the data from the current file to the overall data list
            all_data.append(file_data)

    # Concatenate the data from all the files into a single ndarray
    data = np.stack(all_data)
    return data

experiment_data = load_experiment_data(target_dir = "results/EMA vs monotone")

