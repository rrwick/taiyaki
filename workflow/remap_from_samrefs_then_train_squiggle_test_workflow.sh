#! /bin/bash -eux
set -o pipefail

# Test workflow from fast5 files with remapping to trained squiggle odel
# This is done with just a few reads so the model
# won't be useful for anything.
# This script must be executed with the current directory being the taiyaki base directory

echo ""
echo "Test of remapping using references extracted from fast5s followed by basecall network training starting"
echo ""



# Execute the whole workflow, extracting references, generating per-read-params and mapped-read files and then training
READ_DIR=test/data/reads
SAM_DIR=test/data/aligner_output
# The |xargs puts spaces rather than newlines between the filenames
SAMFILES=$(ls ${SAM_DIR}/*.sam |xargs)
REFERENCEFILE=test/data/genomic_reference.fasta
PREDICT_SQUIGGLE_TEST_FASTA=test/data/phiX174.fasta

echo "SAMFILES=${SAMFILES}"
echo "REAFERENCEFILE=${REFERENCEFILE}"
echo "PREDICT_SQUIGGLE_TEST_FASTA=${PREDICT_SQUIGGLE_TEST_FASTA}"

TAIYAKI_DIR=`pwd`
RESULT_DIR=${TAIYAKI_DIR}/RESULTS/squiggletrain_remap_samref

rm -rf $RESULT_DIR
rm -rf ${TAIYAKI_DIR}/RESULTS/training_ingredients

#TAIYAKIACTIVATE=(nothing) makes the test run without activating the venv at each step. Necessary for running on the git server.
make -f workflow/Makefile \
	MAXREADS=10 \
	READDIR=${READ_DIR} \
	TAIYAKI_ROOT=${TAIYAKI_DIR} \
	DEVICE=cpu \
	MAX_TRAINING_ITERS=2 \
	BAMFILE="${SAMFILES}" \
	REFERENCEFILE=${REFERENCEFILE} \
	PREDICT_SQUIGGLE_TEST_FASTA=${PREDICT_SQUIGGLE_TEST_FASTA} \
	SEED=1 \
	TAIYAKIACTIVATE= \
	squigglepredict_remap_samref


# Check that training chunk log and training log exist and have enough rows for us to be sure something useful has happened



chunklog_lines=`wc -l ${RESULT_DIR}/chunklog.tsv | cut -f1 -d' '`
echo "Number of lines in training chunk log: ${chunklog_lines}"
if [ "$chunklog_lines" -lt "20" ]
then
    echo "Training chunk log too short - not enough chunks generated"
    exit 1
fi

traininglog_lines=`wc -l ${RESULT_DIR}/model.log | cut -f1 -d' '`
echo "Number of lines in training log: ${traininglog_lines}"
if [ "$traininglog_lines" -lt "9" ]
then
    echo "Training log too short- training not started properly"
    exit 1
fi

echo ""
echo "Test of remapping using references extracted from fast5s followed by squiggle network training completed successfully"
echo ""
