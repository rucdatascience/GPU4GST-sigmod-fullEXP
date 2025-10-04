# sh sh/example.sh
cd code/D-PrunedDP++
mkdir build
cd build
cmake ..
make 
./bin/D-PrunedDP++ 2 ../../../data/ Twitch 3 4 0 10

cd ../../../code/D-TrimCDP
mkdir build
cd build
cmake ..
make 
./bin/D-TrimCDP 2 ../../../data/ Twitch 3 4 0 10

cd ../../../code/D-TrimCDP-multi-core-CPU
mkdir build
cd build
cmake ..
make
./bin/D-PrunedDP++ 2 ../../../data/  Twitch  3 4 0 10

cd ../../../code/D-TrimCDP-WB
mkdir build
cd build
cmake ..
make 
./bin/D-TrimCDP-WB 1 ../../../data/ Twitch 3 4 0 10

cd ../../../code/D-TrimCDP-WB-kernel_fusion-no_shared_memory_prefix_scan-global_memory_coalescing
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Twitch  3 4 0 10

cd ../../../code/D-TrimCDP-WB-kernel_fusion-shared_memory_prefix_scan-global_memory_coalescing
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Twitch  3 4 0 10

cd ../../../code/D-TrimCDP-WB-no_global_memory_coalescing
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Twitch  3 4 0 10

cd ../../../code/D-TrimCDP-WB-no_kernel_fusion
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Twitch  3 4 0 10

cd ../../../code/D-TrimCDP-WB-no_virtual_split
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Twitch  3 4 0 10

cd ../../../code/PrunedDP++
mkdir build
cd build
cmake ..
make 
./bin/PrunedDP++ 1 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP
mkdir build
cd build
cmake ..
make 
./bin/TrimCDP 2 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP-multi-core-CPU
mkdir build
cd build
cmake ..
make
./bin/PrunedDP++ 1 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP-WB
mkdir build
cd build
cmake ..
make 
./bin/TrimCDP-WB 1 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP-WB-kernel_fusion-no_shared_memory_prefix_scan-global_memory_coalescing
mkdir build
cd build
cmake .. 
make
./bin/TrimCDP-WB 2 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP-WB-kernel_fusion-shared_memory_prefix_scan-global_memory_coalescing
mkdir build
cd build
cmake .. 
make
./bin/TrimCDP-WB 2 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP-WB-no_global_memory_coalescing
mkdir build
cd build
cmake .. 
make
./bin/TrimCDP-WB 2 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP-WB-no_kernel_fusion
mkdir build
cd build
cmake .. 
make
./bin/TrimCDP-WB 2 ../../../data/ Twitch 3 0 10

cd ../../../code/TrimCDP-WB-no_virtual_split
mkdir build
cd build
cmake .. 
make
./bin/TrimCDP-WB 2 ../../../data/ Twitch 3 0 10