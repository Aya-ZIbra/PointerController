# Inputs
INPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
nq=$4

# The default path for the job is the user's home directory,
#  change directory to where the files are.
cd $PBS_O_WORKDIR


python3 demo.py -m models -i $INPUT_FILE \
                            -FP $FP_MODEL \
                            -d $DEVICE\
                            -nq $nq\
                            -v
