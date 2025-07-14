
cd /home/lijiayu/gst6/code/D-TrimCDP-WB
mkdir build
cd build
cmake ..
make
./bin/D-TrimCDP-WB 2 ../../../data/  Musae  5 3 0 10



cd /home/lijiayu/gst6/code/TrimCDP-WB
mkdir build
cd build
cmake .. 
make
#exe type path data_name T task_start_num(from 0) task_end_num


./bin/TrimCDP-WB 2 ../../../data/ Musae 3 0 10

