"""
Assemble Audio - Rémi Ancay - 16/07/2025

This script implements audio processing functions to assemble and split audio clips from a dataset.

It includes functions to:
1. Assemble or split audio clips into segments of a fixed length.
2. Split the dataset into training and testing sets.
3. Remove classes with fewer than a specified number of files in the test and train sets.
"""
import os
import librosa
import soundfile as sf
import numpy as np
import shutil
import random


# Assemble or split audio clips from the dataset into segments of a fixed length.
# input_dir: Directory containing the audio files organized by species.
# output_dir: Directory to save the audio segments.
# segment_length: Length of each audio segment in seconds.
# keep_remaining: If True, adds padding to the last segment and saves it.
def assemble_split_audio(input_dir="dataset", output_dir="augmented_dataset", segment_length=6, keep_remaining=True):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each species directory in the input directory
    for species in os.listdir(input_dir):
        print(f"Assembling audio for {species}")
        species_path = os.path.join(input_dir, species)
        output_species_path = os.path.join(output_dir, species)

        if not os.path.isdir(species_path):
            continue

        os.makedirs(output_species_path, exist_ok=True)

        audio_files = [f for f in os.listdir(species_path) if f.endswith(".wav")]
        audio_files.sort()

        all_audio = []
        sample_rate = None

        # Load all audio files for the species
        for file in audio_files:
            file_path = os.path.join(species_path, file)
            audio, sr = librosa.load(file_path, sr=None)

            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                raise ValueError(f"Inconsistent sample rates in files of {species} : {sample_rate} vs {sr} for {file}")

            all_audio.append(audio)

        if not all_audio:
            continue # Skip if no audio files found

        # Combine all audio files in the folder
        full_audio = np.concatenate(all_audio)

        # Split into fixed-length segments
        segment_samples = int(segment_length * sample_rate)
        total_segments = len(full_audio) // segment_samples

        for i in range(total_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = full_audio[start:end]

            output_file = os.path.join(output_species_path, f"combined_{i+1}.wav")
            sf.write(output_file, segment, sample_rate)
            print(f"Saved {output_file}")

        # If remaining audio at the end, pad it to the desired length
        remaining = len(full_audio) % segment_samples
        if remaining > 0 and keep_remaining:
            last_segment = librosa.util.fix_length(full_audio[-remaining:], size=segment_samples)
            output_file = os.path.join(output_species_path, f"combined_{total_segments + 1}.wav")
            sf.write(output_file, last_segment, sample_rate)
            print(f"Saved {output_file} (padded)")


# Split the dataset into training and testing sets.
# input_dir: Directory containing the audio files organized by species.
# output_dir: Directory to save the split datasets.
# train_ratio: Proportion of the dataset to include in the training set.
# seed: Random seed for reproducibility.
def split_dataset(input_dir="augmented_dataset", output_dir="split_dataset", train_ratio=0.8, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through each species directory in the input directory
    for species in os.listdir(input_dir):
        species_path = os.path.join(input_dir, species)
        if not os.path.isdir(species_path):
            continue

        print(f"Splitting data for {species}")
        files = [f for f in os.listdir(species_path) if f.endswith(".wav")]
        random.shuffle(files)

        split_index = int(len(files) * train_ratio)
        train_files = files[:split_index]
        test_files = files[split_index:]

        # Create directories for train and test sets for the species
        train_species_path = os.path.join(train_dir, species)
        test_species_path = os.path.join(test_dir, species)
        os.makedirs(train_species_path, exist_ok=True)
        os.makedirs(test_species_path, exist_ok=True)

        for f in train_files:
            shutil.copy(os.path.join(species_path, f), os.path.join(train_species_path, f))

        for f in test_files:
            shutil.copy(os.path.join(species_path, f), os.path.join(test_species_path, f))

        print(f" -> {len(train_files)} fichiers dans train/{species}")
        print(f" -> {len(test_files)} fichiers dans test/{species}")


# Remove classes with fewer than a specified number of files in the test and train sets.
# input_dir: Directory containing the audio files organized by classes.
# min_files: Minimum number of files required for a class to be considered usable.
# rename_them: If True, renames the directories classes with number from 1 to N.
def remove_unusable_classes(input_dir="augmented_dataset", min_files=5, rename_them=True, base_name="class"):
    test_dir = os.path.join(input_dir, "test")
    train_dir = os.path.join(input_dir, "train")

    print("Removing unusable classes that have fewer than", min_files, "files in test and train sets.")

    for classes in os.listdir(test_dir):
        classes_test_path = os.path.join(test_dir, classes)
        classes_train_path = os.path.join(train_dir, classes)

        if not os.path.isdir(classes_test_path):
            continue

        # Load all audio files for the species
        test_files = [f for f in os.listdir(classes_test_path) if f.endswith(".wav")]
        train_files = [f for f in os.listdir(classes_train_path) if f.endswith(".wav")]

        # Check if the number of files in test and train sets is below the minimum threshold
        if len(test_files) < min_files or len(train_files) < min_files:
            print(f"Removing {classes} due to insufficient files: test = {len(test_files)}, train = {len(train_files)}")
            shutil.rmtree(classes_test_path)
            shutil.rmtree(classes_train_path)
        else:
            print(f"{classes} has sufficient files: test = {len(test_files)}, train = {len(train_files)}")

    # If renaming is requested, rename the classes in the test and train directories
    if rename_them:
        print("Renaming classes...")
        test_classes = sorted(os.listdir(test_dir))
        train_classes = sorted(os.listdir(train_dir))

        for i, classes in enumerate(test_classes):
            old_path = os.path.join(test_dir, classes)
            new_path = os.path.join(test_dir, base_name + "_" + str(i + 1))
            os.rename(old_path, new_path)

        for i, classes in enumerate(train_classes):
            old_path = os.path.join(train_dir, classes)
            new_path = os.path.join(train_dir, base_name + "_" + str(i + 1))
            os.rename(old_path, new_path)

        print("Classes renamed successfully from", base_name + "_1", "to", base_name + "_" + str(len(test_classes)))



if __name__ == "__main__":
    # Example usage:
    #split_dataset(input_dir="Datasets/BiggerDataset/CropedBiggerDataset", output_dir="Datasets/BiggerDataset/SplitDataset", train_ratio=0.8, seed=32546789)
    #assemble_split_audio(input_dir="Datasets/BiggerDataset/SplitDataset/train", output_dir="Datasets/BiggerDataset/Dataset/train", segment_length=6, keep_remaining=False)
    #assemble_split_audio(input_dir="Datasets/BiggerDataset/SplitDataset/test", output_dir="Datasets/BiggerDataset/Dataset/test", segment_length=6, keep_remaining=False)

    assemble_split_audio(input_dir="Codes/Datasets/RawDownload/barkopedia_individual_datasets/train", output_dir="Codes/Datasets/BarkopediaIndividualDataset/train", segment_length=6, keep_remaining=False)
    assemble_split_audio(input_dir="Codes/Datasets/RawDownload/barkopedia_individual_datasets/validation", output_dir="Codes/Datasets/BarkopediaIndividualDataset/test", segment_length=6, keep_remaining=False)
    remove_unusable_classes(input_dir="Codes/Datasets/BarkopediaIndividualDataset", min_files=4, rename_them=True, base_name="dog")