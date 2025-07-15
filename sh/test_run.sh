

cd /home/lijiayu/gst6/code/TrimCDP
mkdir build
cd build
cmake .. 
make
#sh sh/run_exp_GPU1_nonHop.sh
#exe type path data_name T task_start_num(from 0) task_end_num

 ./bin/TrimCDP 2 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/D-TrimCDP
mkdir build
cd build
cmake ..
make
#exe type path data_name T  D task_start_num(from 0) task_end_num

 ./bin/D-TrimCDP 2 ../../../data/  Musae  5 3 0 10

cd /home/lijiayu/gst6/code/PrunedDP++
mkdir build
cd build
cmake ..
make
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/PrunedDP++ 1 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/D-PrunedDP++
mkdir build
cd build
cmake ..
make
#exe type path data_name T  D task_start_num(from 0) task_end_num
./bin/D-PrunedDP++ 2 ../../../data/  Musae  5 3 0 10

cd /home/lijiayu/gst6/code/TrimCDP-WB
mkdir build
cd build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/TrimCDP-WB 2 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/D-TrimCDP-WB
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Musae  5 3 0 10

cd /home/lijiayu/gst6/code/TrimCDP-WB-kernel_fusion-shared_memory-coalescing-Kogge_Stone
mkdir build
cd build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/TrimCDP-WB 2 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/TrimCDP-WB-no_kernel_fusion-shared_memory-coalescing-Kogge_Stone
mkdir build
cd build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/TrimCDP-WB 2 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/TrimCDP-WB-kernel_fusion-no_shared_memory-coalescing-Kogge_Stone
mkdir build
cd build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/TrimCDP-WB 2 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/TrimCDP-WB-kernel_fusion-no_shared_memory-no_coalescing-no_Kogge_Stone
mkdir build
cd build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/TrimCDP-WB 2 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/TrimCDP-WB-no_kernel_fusion-no_shared_memory-coalescing-no_Kogge_Stone
mkdir build
cd build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num
./bin/TrimCDP-WB 2 ../../../data/ Musae 3 0 10

cd /home/lijiayu/gst6/code/D-TrimCDP-WB-kernel_fusion-shared_memory-coalescing-Kogge-Stone
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Musae  5 3 0 10

cd /home/lijiayu/gst6/code/D-TrimCDP-WB-no_kernel_fusion-shared_memory-coalescing-Kogge_Stone
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Musae  5 3 0 10


cd /home/lijiayu/gst6/code/D-TrimCDP-WB-kernel_fusion-no_shared_memory-no_coalescing-no_Kogge_Stone
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Musae  5 3 0 10

cd /home/lijiayu/gst6/code/D-TrimCDP-WB-no_kernel_fusion-no_shared_memory-coalescing-no_Kogge_Stone
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Musae  5 3 0 10


cd /home/lijiayu/gst6/code/D-TrimCDP-WB-kernel_fusion-no_shared_memory-coalescing-Kogge_Stone
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Musae  5 3 0 10