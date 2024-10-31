import pysam
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import os

# Define paths
reference_genome_path = "/new-home/25506218/Project/trypano/yeast/SRR_81_results/S288C_reference_sequence_R64-4-1_20230830.fsa"
bam_file_path = "/new-home/25506218/Project/trypano/yeast/SRR_79_results/SRR1924279.sorted.bam"
output_sequences_file = "output/extended_sequences.fasta"
output_csv_file = "output/central_nucleosome_positions.csv"
output_histogram_dir = "output/histograms/"
genome_wide_plot = "output/genome_wide_nucleosome_plot.png"

# Load reference genome
reference_genome = pysam.FastaFile(reference_genome_path)

# Create directories if they don't exist
os.makedirs(output_histogram_dir, exist_ok=True)

# Dictionary to store nucleosome counts per chromosome
chromosome_bins = defaultdict(lambda: defaultdict(int))
csv_data = []
fasta_sequences = []

# Function to process each read
def process_read(read):
    fragment_length = abs(read.template_length)

    # Only considering fragment lengths between 150 and 170 bp
    if 150 <= fragment_length <= 170:
        start = read.reference_start

        # Calculate the central nucleotide
        if fragment_length % 2 == 0:
            central_nucleotide = start + (fragment_length // 2) - 1
        else:
            central_nucleotide = start + (fragment_length // 2)

        # Shift the central position by 1 bp to the left
        shifted_central_nucleotide = central_nucleotide - 1

        # Define a margin (80 bp upstream and 80 bp downstream)
        margin = 80
        start_seq = shifted_central_nucleotide - margin
        end_seq = shifted_central_nucleotide + margin

        # Get chromosome length
        chromosome_length = reference_genome.get_reference_length(read.reference_name)

        # Ensure the sequence isn't too close to the chromosome boundaries
        if start_seq >= 0 and end_seq < chromosome_length:
            # Update the bin for the shifted central nucleotide
            chromosome_bins[read.reference_name][shifted_central_nucleotide] += 1

            # Save this information for the CSV file
            csv_data.append([read.reference_name, shifted_central_nucleotide, chromosome_bins[read.reference_name][shifted_central_nucleotide]])

            # Fetch the sequence from the reference genome (161 bp total: 80 bp upstream + shifted central position + 80 bp downstream)
            sequence = reference_genome.fetch(read.reference_name, start_seq, end_seq + 1)

            # Collect the sequence for batch writing to the FASTA file
            fasta_sequences.append(f">{read.reference_name}:{shifted_central_nucleotide}\n{sequence}\n")

# Function to generate histograms for each chromosome
def plot_histogram(chromosome_data, chromosome_name, chromosome_length):
    output_path = f"{output_histogram_dir}{chromosome_name}_nucleosome_histogram.png"
    plt.figure(figsize=(10, 6))
    positions, counts = zip(*sorted(chromosome_data.items()))
    plt.bar(positions, counts, edgecolor='black')
    plt.title(f'Nucleosome Count Distribution for Chromosome {chromosome_name}')
    plt.xlabel(f'Position on Chromosome {chromosome_name} (bp)')
    plt.ylabel('Nucleosome Count')
    plt.xlim(0, chromosome_length)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Function to save CSV data
def save_csv():
    with open(output_csv_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Chromosome", "Central Position", "Nucleosome Count"])
        csv_writer.writerows(csv_data)

# Function to save FASTA data
def save_fasta():
    with open(output_sequences_file, "w") as seq_file:
        seq_file.writelines(fasta_sequences)

# Function to generate a genome-wide plot
def plot_genome_wide():
    plt.figure(figsize=(12, 8))
    for chromosome_name, chromosome_data in chromosome_bins.items():
        positions, counts = zip(*sorted(chromosome_data.items()))
        plt.plot(positions, counts, label=f"{chromosome_name}")

    plt.title('Genome-wide Nucleosome Count Distribution')
    plt.xlabel('Central Nucleotide Position (bp)')
    plt.ylabel('Nucleosome Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(genome_wide_plot)
    plt.close()

# Main processing function
def process_bam_file(bam_file_path):
    bamfile = pysam.AlignmentFile(bam_file_path, "rb")
    for read in bamfile.fetch():
        if not read.is_unmapped and read.is_proper_pair and not read.is_reverse:
            process_read(read)
    bamfile.close()

    # Save results
    save_fasta()
    save_csv()

    # Generate histograms for each chromosome
    for chromosome_name, chromosome_data in chromosome_bins.items():
        chromosome_length = reference_genome.get_reference_length(chromosome_name)
        plot_histogram(chromosome_data, chromosome_name, chromosome_length)

    # Generate genome-wide plot
    plot_genome_wide()

# Run the main process
if __name__ == "__main__":
    process_bam_file(bam_file_path)
    print("Processing complete.")
