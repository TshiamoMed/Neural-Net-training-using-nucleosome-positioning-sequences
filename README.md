# Neural-Net-training-using-nucleosome-positioning-sequences
Training a neural network using nucleosome positioning sequences and applying to S.cerevisiae , and T.brucei


Overview:

This research project aims to develop a neural network-based approach for predicting nucleosome positioning in two distinct eukaryotic organisms: Saccharomyces cerevisiae (budding yeast) and Trypanosoma brucei (protozoan parasite). The study leverages MNase-Seq data to explore nucleosome distribution patterns across different genomic architectures.


Research Motivation:

Nucleosome positioning plays a crucial role in gene regulation and chromatin structure. Despite advances in high-throughput sequencing, predicting nucleosome occupancy remains challenging, especially across diverse organisms. This project demonstrates the potential of deep learning in capturing complex nucleosome positioning patterns.


# Key Features:

Comparative analysis of nucleosome positioning in S. cerevisiae and T. brucei

Deep learning neural network with skip connections

Advanced data preprocessing techniques

Multiple regularization strategies

Comprehensive performance evaluation



# Prerequisites:
System Requirements:

High-performance computing cluster

Minimum 700GB RAM

16 CPU cores

Python 3.12.0

GPU support recommended



# Required Python Packages:

trim_galore/0.4.1,
Samtools,
bowtie2/2.5.3,
numpy,
pandas,
scikit-learn,
tensorflow/2.9.2,
keras,
matplotlib,
seaborn,
biopython.


# Project Pipeline:

1. Data Preprocessing:
   
Script: trim.pbs

*Uses Trim Galore for quality control,

*Adapter trimming with strict parameters.

*Ensures high-quality sequencing reads


2. Genome Alignment:
   
*Script: align.py

*Uses Bowtie2 for efficient paired-end read alignment

*Post-alignment processing with SAMtools


3. Nucleosome Position Extraction:
   
*Script: nucleosome_extraction.py

*Filters fragments between 150-170bp

*Calculates central positions

*Extracts 161bp sequences (80bp upstream and downstream)

*Bins nucleosome counts


4. Binary encoding:

script: one_hot_encode.py

This scripts compresses the extracted sequences into 4 bits one hot encoding.


5. Feature Engineering:
    
*Script: baum.py

*Baum-Welch algorithm for hidden state detection

*Feature concatenation


6. Data Splitting:
   
*Script: split.py

*80% training, 10% validation, 10% testing

*Separate splits for for different experimental samples.


7. Neural Network Training:
   
*Script: network_training_yeast.py and network_training_T_brucei.py

*Custom neural network architecture

*Skip connections

*Regularization techniques

*Custom loss function

*Log transformation and standardscaler for yeast target values

*MinMaxscaler for T.brucei target values


8. Model Evaluation:
    
*Script: yeast_model_evaluation.py and T_brucei_model_evaluation.py

*Performance metrics calculation

*MSE, MAE, R-squared analysis

*Visualization of prediction accuracy


# Datasets:

S. cerevisiae: GEO NCBI - GSE67148, 
T. brucei: GEO NCBI - GSE90593

# Reference Genomes:

S. cerevisiae: S288C_reference_sequence_R64-4-1_20230830.fsa. 
T. brucei: TriTrypDB-68_TbruceiLister427_2018_Genome.fasta

# Key Findings:

Higher prediction accuracy in quiescent cells. Challenges with data imbalance. Unique challenges in T. brucei nucleosome positioning

# Limitations and Future Work:

Improve handling of high nucleosome count regions. Incorporate additional genomic features. Develop advanced data augmentation techniques

Reproducibility
All scripts, trained models, and additional resources are available in this repository. Detailed documentation and comments are provided in each script.


