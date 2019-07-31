#!/usr/bin/env python3
import argparse
import pysam
import sys

from taiyaki.bio import reverse_complement
from taiyaki.cmdargs import proportion, FileExists
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.helpers import fasta_file_to_dict, open_file_or_stdout


parser = argparse.ArgumentParser(
    description='Extract reference sequence for each read from a SAM alignment file',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_common_command_args(parser, ["output"])

parser.add_argument('--min_coverage', metavar='proportion', default=0.6, type=proportion,
                    help='Ignore reads with alignments shorter than min_coverage * read length')
parser.add_argument('--pad', type=int, default=0,
                    help='Number of bases by which to pad reference sequence')
parser.add_argument('reference', action=FileExists,
                    help="Genomic references that reads were aligned against")
parser.add_argument('input', metavar='input.sam', nargs='+',
                    help="SAM or BAM file(s) containing read alignments to reference")

STRAND = {0: '+',
          16: '-'}


def get_refs(sam, ref_seq_dict, min_coverage=0.6, pad=0):
    """Read alignments from sam file and return accuracy metrics
    """
    with pysam.Samfile(sam, 'r') as sf:
        for read in sf:
            if read.flag != 0 and read.flag != 16:
                continue

            coverage = float(read.query_alignment_length) / read.query_length
            if coverage < min_coverage:
                continue

            read_ref = ref_seq_dict.get(sf.references[read.reference_id], None)
            if read_ref is None:
                continue

            start = max(0, read.reference_start - pad)
            end = min(len(read_ref), read.reference_end + pad)

            strand = STRAND[read.flag]
            read_ref = read_ref.decode() if isinstance(read_ref, bytes) else read_ref

            if strand == "+":
                read_ref = read_ref[start:end].upper()
            else:
                read_ref = reverse_complement(read_ref[start:end].upper())

            fasta = ">{}\n{}\n".format(read.qname, read_ref)

            yield (read.qname, fasta)


def main():
    args = parser.parse_args()

    sys.stderr.write("* Loading references (this may take a while for large genomes)\n")
    references = fasta_file_to_dict(args.reference, allow_N=True)

    sys.stderr.write("* Extracting read references using SAM alignment\n")
    with open_file_or_stdout(args.output) as fh:
        for samfile in args.input:
            for (name, fasta) in get_refs(samfile, references, args.min_coverage, args.pad):
                fh.write(fasta)


if __name__ == '__main__':
    main()
