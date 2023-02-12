import shutil


def compress_folder(output_filename, input_folder_name):
    shutil.make_archive(output_filename, 'zip', input_folder_name)


if __name__ == "__main__":
    compress_folder("training_data", "./training_data/raw")
