#!/bin/bash

# The job name is used to determine the name of job output and error files
#SBATCH -J exp2.DD2427

# Set the time allocation to be charged
#SBATCH -A edu16.SF2568

# Request a mail when the job starts and ends
#SBATCH --mail-type=ALL

# Maximum job elapsed time should be indicated whenever possible
#SBATCH -t 13:00:00

# Number of nodes that will be reserved for a given job
#SBATCH --nodes=1


#SBATCH -e error.log
#SBATCH -o output.log

#SBATCH --mem=2000000

export LOCAL_ANACONDA=/cfs/klemming/nobackup/b/bened/hannes/dd2427
source $LOCAL_ANACONDA/bin/activate $LOCAL_ANACONDA
python test_conv_net_feed.py > output.out
source $LOCAL_ANACONDA/bin/deactivate
