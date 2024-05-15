#!/bin/bash -l

echo mach1.5
cd /home/watanabe/Projects/outputs/Sod_mach1.5
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach1.5
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach1.5/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach1.5
echo mach2.0
cd /home/watanabe/Projects/outputs/Sod_mach2.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach2.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach2.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach2.0
echo mach5.0
cd /home/watanabe/Projects/outputs/Sod_mach5.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach5.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach5.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach5.0
echo mach10.0
cd /home/watanabe/Projects/outputs/Sod_mach10.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach10.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach10.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach10.0
echo mach25.0
cd /home/watanabe/Projects/outputs/Sod_mach25.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach25.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach25.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach25.0
echo mach50.0
cd /home/watanabe/Projects/outputs/Sod_mach50.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach50.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach50.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach50.0
echo mach100.0
cd /home/watanabe/Projects/outputs/Sod_mach100.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach100.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach100.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach100.0
echo mach250.0
cd /home/watanabe/Projects/outputs/Sod_mach250.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach250.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach250.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach250.0
echo mach500.0
cd /home/watanabe/Projects/outputs/Sod_mach500.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach500.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach500.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach500.0
echo mach1000.0
cd /home/watanabe/Projects/outputs/Sod_mach1000.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_mach1000.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_mach1000.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_mach1000.0
