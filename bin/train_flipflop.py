#!/usr/bin/env python3
import argparse
from collections import defaultdict
import gc
import numpy as np
import os
import random
from shutil import copyfile
import sys
import time

import torch
import taiyaki.common_cmdargs as common_cmdargs
from taiyaki.cmdargs import (FileExists, FilesExist, NonNegative, Positive, proportion)

from taiyaki import chunk_selection, ctc, flipflopfings, helpers, mapped_signal_files, variables
from taiyaki import __version__


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Train a flip-flop neural network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

common_cmdargs.add_common_command_args(parser, """adam chunk_logging_threshold device filter_max_dwell filter_mean_dwell
                                                  limit lrdecay niteration overwrite quiet save_every
                                                  sample_nreads_before_filtering version weight_decay""".split())

parser.add_argument('--min_batch_size', default=50, metavar='chunks', type=Positive(int),
                    help='Number of chunks to run in parallel for chunk_len = chunk_len_max.' +
                         'Actual batch size used is (min_batch_size / chunk_len) * chunk_len_max')
parser.add_argument('--chunk_len_min', default=2000, metavar='samples', type=Positive(int),
                    help='Min length of each chunk in samples (chunk lengths are random between min and max)')
parser.add_argument('--chunk_len_max', default=4000, metavar='samples', type=Positive(int),
                    help='Max length of each chunk in samples (chunk lengths are random between min and max)')

parser.add_argument('--input_strand_list', default=None, action=FileExists,
                    help='Strand summary file containing column read_id. Filenames in file are ignored.')
parser.add_argument('--min_prob', default=1e-30, metavar='p', type=proportion,
                    help='Minimum probability allowed for training')
parser.add_argument('--seed', default=None, metavar='integer', type=Positive(int),
                    help='Set random number seed')
parser.add_argument('--sharpen', default=1.0, metavar='factor',
                    type=Positive(float), help='Sharpening factor')
parser.add_argument('--smooth', default=0.45, metavar='factor', type=proportion,
                    help='Smoothing factor for reporting progress')
parser.add_argument('--stride', default=2, metavar='samples', type=Positive(int),
                    help='Stride for model')
parser.add_argument('--winlen', default=19, type=Positive(int),
                    help='Length of window over data')

parser.add_argument('--reload_after_batches', default=500, type=Positive(int),
                    help='Reload training data after this many batches (only used if --limit is set)')

parser.add_argument('model', action=FileExists,
                    help='File to read python model description from')
parser.add_argument('output', help='Prefix for output files')
parser.add_argument('input', action=FilesExist, nargs='+',
                    help='files containing mapped reads')


def save_model(network, output, index=None):
    if index is None:
        basename = 'model_final'
    else:
        basename = 'model_checkpoint_{:05d}'.format(index)

    model_file = os.path.join(output, basename + '.checkpoint')
    torch.save(network, model_file)
    params_file = os.path.join(output, basename + '.params')
    torch.save(network.state_dict(), params_file)


def load_read_data(input_files, read_limit):
    read_data = []
    for input_file in input_files:
        log.write('* Loading data from {}\n'.format(input_file))
        log.write('* Per read file MD5 {}\n'.format(helpers.file_md5(input_file)))
        with mapped_signal_files.HDF5(input_file, "r") as per_read_file:
            read_data += per_read_file.get_multiple_reads(read_ids, max_reads=read_limit)
            # read_data now contains a list of reads
            # (each an instance of the Read class defined in mapped_signal_files.py, based on dict)
    random.shuffle(read_data)
    return read_data


if __name__ == '__main__':
    args = parser.parse_args()

    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not args.overwrite:
        sys.stderr.write('Error: Output directory {} exists but --overwrite is false\n'.format(args.output))
        exit(1)
    if not os.path.isdir(args.output):
        sys.stderr.write('Error: Output location {} is not directory\n'.format(args.output))
        exit(1)

    copyfile(args.model, os.path.join(args.output, 'model.py'))

    # Create a logging file to save details of chunks.
    # If args.chunk_logging_threshold is set to 0 then we log all chunks including those rejected.
    chunk_log = chunk_selection.ChunkLog(args.output)

    log = helpers.Logger(os.path.join(args.output, 'model.log'), args.quiet)
    log.write('* Taiyaki version {}\n'.format(__version__))
    log.write('* Command line\n')
    log.write(' '.join(sys.argv) + '\n')

    if args.input_strand_list is not None:
        read_ids = list(set(helpers.get_read_ids(args.input_strand_list)))
        log.write('* Will train from a subset of {} strands, determined by read_ids in input strand list\n'.format(len(read_ids)))
    else:
        log.write('* Will train from all strands\n')
        read_ids = 'all'

    if args.limit is not None:
        log.write('* Limiting number of strands to {} per file\n'.format(args.limit))

    read_data = load_read_data(args.input, args.limit)
    log.write('* Loaded {} reads.\n'.format(len(read_data)))

    # Get parameters for filtering by sampling a subset of the reads
    # Result is a tuple median mean_dwell, mad mean_dwell
    # Choose a chunk length in the middle of the range for this
    sampling_chunk_len = (args.chunk_len_min + args.chunk_len_max) // 2
    filter_parameters = chunk_selection.sample_filter_parameters(read_data,
                                                                 args.sample_nreads_before_filtering,
                                                                 sampling_chunk_len,
                                                                 args,
                                                                 log,
                                                                 chunk_log=chunk_log)

    medmd, madmd = filter_parameters
    
    log.write("* Sampled {} chunks: median(mean_dwell)={:.2f}, mad(mean_dwell)={:.2f}\n".format(
              args.sample_nreads_before_filtering, medmd, madmd))
    log.write('* Reading network from {}\n'.format(args.model))
    nbase = len(read_data[0]['alphabet'])
    model_kwargs = {
        'stride': args.stride,
        'winlen': args.winlen,
        'insize': 1,  # Number of input features to model e.g. was >1 for event-based models (level, std, dwell)
        'outsize': variables.nstate_flipflop(nbase)
    }
    network = helpers.load_model(args.model, **model_kwargs).to(device)
    log.write('* Network has {} parameters.\n'.format(sum([p.nelement()
                                                           for p in network.parameters()])))

    learning_rate = args.adam.rate
    betas = args.adam[1:]
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,
                                 betas=betas, weight_decay=args.weight_decay)
    lr_decay = lambda step: args.lrdecay / (args.lrdecay + step)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_decay)

    score_smoothed = helpers.ExponentialSmoother(args.smooth)

    log.write('* Dumping initial model\n')
    save_model(network, args.output, 0)

    total_bases = 0
    total_samples = 0
    total_chunks = 0
    rejection_dict = defaultdict(lambda : 0)  # To count the numbers of different sorts of chunk rejection

    t0 = time.time()
    log.write('* Training\n')
    


    for i in range(args.niteration):
        lr_scheduler.step()
        # Chunk length is chosen randomly in the range given but forced to be a multiple of the stride
        batch_chunk_len = (np.random.randint(args.chunk_len_min, args.chunk_len_max + 1) // args.stride) * args.stride
        # We choose the batch size so that the size of the data in the batch is about the same as
        # args.min_batch_size chunks of length args.chunk_len_max
        target_batch_size = int(args.min_batch_size * args.chunk_len_max / batch_chunk_len + 0.5)
        # ...but it can't be more than the number of reads.
        batch_size = min(target_batch_size, len(read_data))


        # If the logging threshold is 0 then we log all chunks, including those rejected, so pass the log
        # object into assemble_batch
        if args.chunk_logging_threshold == 0:
            log_rejected_chunks = chunk_log
        else:
            log_rejected_chunks = None
        # Chunk_batch is a list of dicts.
        chunk_batch, batch_rejections = chunk_selection.assemble_batch(read_data, batch_size, batch_chunk_len,
                                                                       filter_parameters, args, log,
                                                                       chunk_log=log_rejected_chunks)
        total_chunks += len(chunk_batch)

        # Update counts of reasons for rejection
        for k, v in batch_rejections.items():
            rejection_dict[k] += v

        # Shape of input tensor must be  (timesteps) x (batch size) x (input channels)
        # in this case                  batch_chunk_len x batch_size x 1
        stacked_current = np.vstack([d['current'] for d in chunk_batch]).T
        indata = torch.tensor(stacked_current, device=device, dtype=torch.float32).unsqueeze(2)
        # Sequence input tensor is just a 1D vector, and so is seqlens
        seqs = torch.tensor(np.concatenate([flipflopfings.flip_flop_code(d['sequence']) for d in chunk_batch]),
                            device=device, dtype=torch.long)
        seqlens = torch.tensor([len(d['sequence']) for d in chunk_batch], dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = network(indata)
        lossvector = ctc.crf_flipflop_loss(outputs, seqs, seqlens, args.sharpen)
        loss = lossvector.sum() / (seqlens > 0.0).float().sum()
        loss.backward()
        optimizer.step()

        fval = float(loss)
        score_smoothed.update(fval)

        # Check for poison chunk and save losses and chunk locations if we're poisoned
        # If args.chunk_logging_threshold set to zero then we log everything
        if fval / score_smoothed.value >= args.chunk_logging_threshold:
            chunk_log.write_batch(i, chunk_batch, lossvector)

        total_bases += int(seqlens.sum())
        total_samples += int(indata.nelement())

        # Doing this deletion leads to less CUDA memory usage.
        del indata, seqs, seqlens, outputs, loss, lossvector
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if (i + 1) % args.save_every == 0:
            save_model(network, args.output, (i + 1) // args.save_every)
            log.write('C')
        else:
            log.write('.')

        if (i + 1) % 50 == 0:
            # In case of super batching, additional functionality must be
            # added here
            learning_rate = lr_scheduler.get_lr()[0]
            tn = time.time()
            dt = tn - t0
            t = ' {:5d} {:6.4f}  {:5.2f}s ({:.2f} ksample/s {:.2f} kbase/s) lr={:.2e}'
            log.write(t.format((i + 1) // 50, score_smoothed.value,
                               dt, total_samples / 1000.0 / dt,
                               total_bases / 1000.0 / dt, learning_rate))
            # Write summary of chunk rejection reasons
            for k, v in rejection_dict.items():
                log.write(" {}:{} ".format(k, v))
            log.write("\n")
            total_bases = 0
            total_samples = 0
            t0 = tn

        if args.limit is not None and (i + 1) % args.reload_after_batches == 0 and i != args.niteration - 1:
            # Periodically reload the training data to get a different random subset.
            read_data = None
            gc.collect()
            read_data = load_read_data(args.input, args.limit)

    save_model(network, args.output)
