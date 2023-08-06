#!/bin/bash

# Get the highest python version installed
python_version=$(ls /usr/bin | grep -E '^python3[0-9]*$' | sort -V | tail -1)

# If no Python version found, exit
if [ -z "$python_version" ]; then
    echo "No suitable Python version found."
    exit 1
fi

METHOD="cce" # default method
EPOCHS=""    # default is not setting epochs (let the python script use its own default)
DATASET="ml-100k"

# Process command-line options
while getopts "m:e:d:" opt; do
    case $opt in
        m) METHOD="$OPTARG";;
        e) EPOCHS="$OPTARG";;
        d) DATASET="$OPTARG";;
        *) echo "Usage: $0 [--method METHOD] [--epochs EPOCHS] [--dataset DATASET]"; exit 1;;
    esac
done

# Define a function to extract the smallest validation loss from output
extract_smallest_loss() {
    local output="$1"
    echo "$output" | grep -o 'Validation Loss: [0-9]\+\.[0-9]\+' | awk -F': ' '{print $2}' | sort -n | head -1
}

declare -a ppds=()   # To store the corresponding ppd values
losses=()            # Flat array to store the smallest loss values for each ppd and seed

script_dir=$(dirname "$0") # Directory where the script is located

# Number of times to run the program
runs=3

# Progress bar variables
total=$((10 * runs))
current=0
bar_length=50 # 50 characters long

print_progress_bar() {
    local percent=$1
    local filled=$((bar_length * current / total))
    local empty=$((bar_length - filled))
    printf 'Progress: [%-*s] %d%%\r' $bar_length $(printf '#%.0s' $(seq 1 $filled)) $percent
}

lo_pow=5
hi_pow=12

# Print an initial empty progress bar
print_progress_bar 0

for run in $(seq 1 $runs); do
    #seed=$((RANDOM % 10000))
    seed=$run
    for i in $(seq $lo_pow $hi_pow); do
        ppd=$((2**$i))
         output=$($python_version "$script_dir/movielens.py" --method $METHOD --ppd $ppd --seed $seed --dataset $DATASET ${EPOCHS:+--epochs $EPOCHS})
        smallest_loss=$(extract_smallest_loss "$output")
        ppds[$i]=$ppd
        index=$(( (run - 1) * 10 + i ))
        losses[$index]=$smallest_loss

        # Update the progress bar
        current=$((current + 1))
        percent=$((100 * current / total))
        print_progress_bar $percent
    done
done

# Move to the next line after the progress bar
echo

# Print results
header="ppd"
for run in $(seq 1 $runs); do
    header+="\tseed_$run"
done
echo -e "$header"
for i in $(seq $lo_pow $hi_pow); do
    line="${ppds[$i]}"
    for run in $(seq 1 $runs); do
        index=$(( (run - 1) * 10 + i ))
        line+="\t${losses[$index]}"
    done
    echo -e "$line"
done
