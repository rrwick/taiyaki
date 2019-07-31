This document describes file formats used in the Taiyaki package.

## Fast5 files

The package reads single or multi-read fast5 files using a wrapper around the **ont_fast5_api** package.

## Strand lists

Strand lists should be tab-separated text files having columns ('filename' or 'filename_fast5' and not both) and / or 'read_id'.
If a strand list file is supplied as an optional argument to a script, then
1. If no column 'read_id' is present, then all files with names in the column 'filename' or 'filename_fast5' are read.
2. If no column 'filename' or 'filename_fast5' is present, then all reads with read_ids in the 'read_id' column are are read from files in the directory specified.
3. If there is a ('filename' or 'filename_fast5') column and a 'read_id' column, then the strand list is regarded as a list of pairs (filename, read_id).

## Per-read parameter files

The script **bin/generate_per_read_parameters.py** creates a tsv file with columns ('UUID', 'trim_start', 'trim_end', 'shift', 'scale') which give instructions for
handling of each read. The shift and scale parameters are chosen so that

    y = (current_in_pA - shift)/scale
    
is standardised: that is, so that roughly, mean(**y**)=0 and std(**y**)=1 (more robust statistics are used by the script to generate the parameters).

    UUID				trim_start	trim_end	shift			scale
    6a8a74ff-5316-41d8-825d-a018af4242bf	200	50	85.43114135742188	15.168887446289057
    906f26ce-367a-4d3c-b279-ca86f6db7255	200	50	97.36762817382814	15.927331818603507
    90b3c72f-ac34-4337-b33a-2fecd0216b99	200	50	82.36786376953125	15.076369731445316

We expect users to find reasons to generate their own per-read-parameter files or to modify the ones generated by this script.

## Reference files

Files to store the reference for each read are used as an ingredient for remapping.

These are fasta files where the comment line for each sequence is the UUID:

    >6a8a74ff-5316-41d8-825d-a018af4242bf
    GTGCTTGTGGGGTATTGCTCAAGAAATTTTTGCCCAGATCAATGTTCTGGAGATTTTACCCAATGT.....
    >906f26ce-367a-4d3c-b279-ca86f6db7255
    AATCCTGCCTCTAAAGAAAGAAAAAAAAAAATCAGCTAGGTGTAGCCATAGGCAGCTGTAGTCCCA.....

## Mapped signal files (v. 8)

Data for training is stored in mapped signal files.
The class **HDF5** in **taiyaiki/mapped_signal_files.py** provides an API for reading and writing these files, and also
methods for checking that a file conforms to the specification.

The files are HDF5 files with the following structure.

    HDF5_file/
      ├── attribute: alphabet (str)
      ├── attribute: collapse_alphabet (str)
      ├── attribute: mod_long_names (str)
      ├── attribute: version (integer)
      └── group: Reads/
          ├── group: <read_id_1>
          ├── group: <read_id_2>
          ├── group: <read_id_3>
          .
          .


Each read_id is a UUID, and the data in each read group is:

|   **name**        |**attribute/dataset** | **type**  | **description**                                                    |
|-------------------|----------------------|-----------|--------------------------------------------------------------------|
| shift_frompA      |  attr                | float     | shift parameter - see 'per-read-parameter files' above             |
| scale_frompA      |  attr                | float     | scale parameter - see 'per-read-parameter files' above             |
| range             |  attr                | float     | see equation below                                                 |
| offset            |  attr                | float     | see equation below                                                 |
| digitisation      |  attr                | float     | see equation below                                                 |
| Dacs              |  dataset             | int16     | signal data representing current through pore (see equation below) |
| Ref_to_signal     |  dataset             | int32     | Ref_to_signal[n] = location in Dacs associated with Reference[n]   |
| Reference         |  dataset             | int16     | alphabet[Reference[n]] is the nth base in the reference sequence   |
| mapping_score     | attr (optional)      | str       | score associated with mapping of ref to signal                     |
| mapping_method    | str (optional)       | str       | short description of mapping method                                |


The current in pA is calculated from the integers in Dacs by the equation

    current = (Dacs + offset ) * range / digitisation
    
    
## Chunk logs

During training, **bin/train_flipflop.py** generates (input,output) pairs of (signal,sequence) for network training.
We refer to each of these (signal,sequence) pairs as a chunk. Some chunks are rejected rather than being fed into the
training loop, either because the required data could not be found or because they are filtered out. For example, we
filter out chunks which contain very long stays (where the molecule appears to be stationary in the pore for a long time) because we
expect them to make training more difficult.

With the option **--chunk_logging_threshold 0**, the scripts **bin/train_flipflop.py** and **bin/train_squiggle.py** produce chunk logs.

These are tab-separated text files which describe the chunks selected and rejected, giving the training loss for each chunk that was
used in training, and a reason for rejection for those chunks which were not used.

Before training starts, 1000 chunks are sampled to determine the baseline for filtering. These chunks are recorded (when **--chunk_logging_threshold 0**) at the start of the
chunk log file, and they can be distinguished from those generated in the training loop because they are not marked as rejected, but do not have a loss associated with them.

The script **misc/plot_chunk_log.py** can be used to plot the data in this file.

## Model files

* Neural network descriptions (with parameters not specified) are needed as an input to training. These are python files: an example is given in the directory **models**.
* It is also possible to use the result of earlier training runs as a starting point: in this case use a **.checkpoint** file (see below). 
* Trained network files are much larger than the python files which define the structure of a network. For example, **bin/train_flipflop.py** saves trained models at each checkpoint and at the end of training in two different formats:
    * **.params** files store the model parameters in a flat pytorch structure.
    * **.checkpoint** files can be used to read a network directly into a pytorch function using **torch.load()**.
    * The script **bin/dump_json.py** transforms a **.checkpoint** file into a **json**-based format which can be used by Guppy.
    * **bin/prepare_mapped_reads.py** needs a trained flip-flop network to use for remapping. This is in the **.checkpoint** format, and an example can be found in the **models** directory.


## Modified base output file

Modified base output scores are output from the **bin/basecall.py** into an HDF5 file with the following format.
More negative modified base scores indicate modified bases and more positive scores indicate canonical bases.
These scores are not calibrated and thus no statistical meaning should be assumed for the scores.

The files are HDF5 files with the following structure:

    HDF5_file/
      ├── dataset: mod_long_names (string)
      └── group: Reads/
          ├── dataset: <read_id_1>
          ├── dataset: <read_id_2>
          ├── dataset: <read_id_3>
          .
          .

Each read_id is a UUID, and each read dataset is of size [basecalls length] x [number of modified bases].
Rows represents the modified base scores for that index within that read's basecalls.
Columns represent scores for the modified base in the order specified in mod_long_names.
Modified base scores are only produced where applicable according to the canonical base associated with each modification (e.g. `5mC` calls are only produced at `C` basecalls).
All other values within the read datasets are `nan`.