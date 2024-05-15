#!/bin/bash -l

# setup FLASH files for SLURM submission

helpFunction()
{
   echo ""
   echo "Usage: $0 -S SIM -F FLASH_DIR -O OUTPUT_DIR -R RUN"
   echo -e "\t-S Name of Simulation"
   echo -e "\t-F Directory to FLASH ISM"
   echo -e "\t-O Parent Directory for SLURM submission"
   echo -e "\t-R Directory Name for SLURM submission"
   exit 1 # Exit script after printing help
}

while getopts "S:F:O:R:P:h:" opt;
do
   case "$opt" in
      S ) SIM="$OPTARG" ;;
      F ) FLASH_DIR="$OPTARG" ;;
      O ) OUTPUT_DIR="$OPTARG" ;;
      R ) RUN="$OPTARG" ;;
      P ) PAR="$OPTARG" ;;
      h ) helpFunction ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# set default values
if [ -z "$SIM" ]; then SIM="RunawayStarISM"; fi
if [ -z "$FLASH_DIR"]; then FLASH_DIR="/home/watanabe/Projects/meerkat_bowshocks/FlashISM"; fi
if [ -z "$OUTPUT_DIR"]; then OUTPUT_DIR="/home/watanabe/Projects/meerkat_bowshocks/outputs"; fi
if [ -z "$RUN"]; then RUN="$SIM"; fi
if [ -z "$PAR"]; then PAR="flash.par"; fi

# setup the FLASH simulation
cd "$FLASH_DIR";

OBJ_DIR="$FLASH_DIR/obj$RUN"
if [ -d $OBJ_DIR ]; 
then
    echo "$OBJ_DIR exists, removing.";
    rm -rf $OBJ_DIR;
fi

mkdir $OBJ_DIR
./setup "$SIM" -auto -3d -objdir=$OBJ_DIR -maxblocks=2000 -nxb=8 -nyb=8 -nzb=8 -site walch01 -debug -unit=diagnostics/ShockFind/ShockFindMain -unit=physics/sourceTerms/CRinjection/CRinjectionMain -unit=physics/sourceTerms/CRdiffusion/CRdiffusionSpectral -unit=physics/sourceTerms/CRsources/CRsourcesMain saveShockFindVars=True saveCRinjectionVars=True MHD=True
cd $OBJ_DIR;
make -j4;

echo "Completed compilation"; 

# make new directory if it already doesnt exist
# otherwise delete it
RUN_DIR="$OUTPUT_DIR/$RUN";
echo "New Directory: $RUN_DIR";

if [ -d $RUN_DIR ]; 
then
    echo "$RUN_DIR exists, removing.";
    rm -rf $RUN_DIR;
fi

mkdir $RUN_DIR;
echo "Made New Directory $RUN_DIR";

# copy all relevant files into RUN_DIR
cp "$FLASH_DIR/source/Simulation/SimulationMain/$SIM/$PAR" "$RUN_DIR";
mv "$RUN_DIR/$PAR" "$RUN_DIR/flash.par"  # rename to flash.par
cp "$OBJ_DIR/flash4" "$RUN_DIR";
cp "$FLASH_DIR/source/Simulation/SimulationMain/$SIM/source_stellar.dat" "$RUN_DIR";
cp "$FLASH_DIR/source/Simulation/SimulationMain/$SIM/source_sinks.dat" "$RUN_DIR";
cp "$FLASH_DIR/source/Simulation/SimulationMain/$SIM/sink_position.txt" "$RUN_DIR";
cp "$FLASH_DIR/source/Simulation/SimulationMain/$SIM/cooltab-plewa.txt" "$RUN_DIR";
cp "./submit.sh" "$RUN_DIR";

echo "Copied all relevant files into $RUN_DIR";

# make logs file to store logs
mkdir "$RUN_DIR/logs";

# print end results
ls $RUN_DIR;





