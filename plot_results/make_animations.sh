#! /bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -S SIM"
   echo -e "\t-S Name of Simulation"
   exit 1 # Exit script after printing help
}

while getopts "S:h:" opt;
do
   case "$opt" in
      S ) SIM="$OPTARG" ;;
      h ) helpFunction ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

data_dir="/home/watanabe/Projects/meerkat_bowshocks/figures/$SIM"
ani_dir="/home/watanabe/Projects/meerkat_bowshocks/figures/animations/$SIM"

if [ ! -d "$ani_dir" ]; then
    mkdir "$ani_dir"
fi

for field_dir in "$data_dir"/*; 
do
    # Check if the current item is a file (not a directory)
    if [ -d "$field_dir" ]; then
        echo "Processing field: $field_dir"
        field=$(basename $field_dir)
        ffmpeg -framerate 10 -pattern_type glob -i "$field_dir/axis2/"$field"_sl_axis2_*.png" $ani_dir/"$field"_sl_axis2_anim.mp4
    fi
done