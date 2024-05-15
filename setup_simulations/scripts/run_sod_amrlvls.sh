#!/bin/bash -l

echo lvl4.0
cd /home/watanabe/Projects/outputs/Sod_CRinjection_lvl4.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl4.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_CRinjection_lvl4.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl4.0
echo lvl5.0
cd /home/watanabe/Projects/outputs/Sod_CRinjection_lvl5.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl5.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_CRinjection_lvl5.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl5.0
echo lvl6.0
cd /home/watanabe/Projects/outputs/Sod_CRinjection_lvl6.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl6.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_CRinjection_lvl6.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl6.0
echo lvl7.0
cd /home/watanabe/Projects/outputs/Sod_CRinjection_lvl7.0
rm -rf *sod* *.dump *.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl7.0
mpirun -np 6 ./flash4 > /home/watanabe/Projects/outputs/Sod_CRinjection_lvl7.0/stdout.log
ls /home/watanabe/Projects/outputs/Sod_CRinjection_lvl7.0
