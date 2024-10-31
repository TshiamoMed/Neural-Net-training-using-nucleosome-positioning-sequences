import numpy as np
from Bio import SeqIO
import os

# One-hot encoding dictionary for DNA bases
one_hot_map = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0]  # For ambiguous bases
}

def sequence_to_one_hot(sequence):
    """
    Converts a DNA sequence into a one-hot encoded numpy array.
    """
    # Convert each base to its one-hot encoding
    one_hot_encoded_seq = [one_hot_map.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(one_hot_encoded_seq)

def fasta_to_one_hot(fasta_file="extended_sequences.fasta", output_file="one_sequences.npy", seq_length=161):
    
    #Converts sequences from a FASTA file to one-hot encoding and saves them as a numpy array.
   
    if not os.path.exists(fasta_file):
        print(f"Error: FASTA file '{fasta_file}' not found.")
        return

    one_hot_sequences = []
    invalid_sequences_count = 0
    
    try:
        # Parse the FASTA file
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence = str(record.seq).upper()
            
            # Check if the sequence length is 161 bp
            if len(sequence) != seq_length:
                invalid_sequences_count += 1
                print(f"Sequence '{record.id}' has length {len(sequence)}; expected {seq_length} bp.")
                continue
            
            # One-hot encode the sequence
            one_hot_encoded_seq = sequence_to_one_hot(sequence)
            one_hot_sequences.append(one_hot_encoded_seq)

        # Convert the list of one-hot encoded sequences into a numpy array
        if one_hot_sequences:  # Check if there are valid sequences to save
            one_hot_sequences = np.array(one_hot_sequences)
            np.save(output_file, one_hot_sequences)
            print(f"One-hot encoded sequences saved to {output_file}")
            print(f"Array shape: {one_hot_sequences.shape}")
        else:
            print("No valid sequences found to encode.")
        
        # Print the count of invalid sequences
        print(f"Total invalid sequences: {invalid_sequences_count}")

    except Exception as e:
        print(f"Error processing file: {e}")

# Run the function
fasta_to_one_hot()