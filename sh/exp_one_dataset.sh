cd code/D-PrunedDP++
mkdir build
cd build
cmake ..
make 
./bin/D-PrunedDP++ 2 ../../../data/ Twitch 3 4 0 299
./bin/D-PrunedDP++ 2 ../../../data/ Twitch 5 4 0 299
./bin/D-PrunedDP++ 2 ../../../data/ Twitch 5 2 0 299
./bin/D-PrunedDP++ 2 ../../../data/ Twitch 5 3 0 299
./bin/D-PrunedDP++ 2 ../../../data/ Twitch 7 4 0 299
#sh sh/exp_one_dataset.sh


cd ../../../code/PrunedDP++
mkdir build
cd build
cmake ..
make 
 ./bin/PrunedDP++ 1 ../../../data/ Twitch 3 0 299
 ./bin/PrunedDP++ 1 ../../../data/ Twitch 5 0 299
 ./bin/PrunedDP++ 1 ../../../data/ Twitch 7 0 299
 

cd ../../../code/GPUGST
mkdir build
cd build
cmake ..
make 
 ./bin/GPUGST 2 ../../../data/ Twitch 3 0 299
 ./bin/GPUGST 2 ../../../data/ Twitch 5 0 299
  ./bin/GPUGST 2 ../../../data/ Twitch 7 0 299


 cd ../../../code/D-GPUGST
mkdir build
cd build
cmake ..
make 
 ./bin/D-GPUGST 2 ../../../data/ Twitch 3 4 0 299
./bin/D-GPUGST 2 ../../../data/ Twitch 5 2 0 299
./bin/D-GPUGST 2 ../../../data/ Twitch 5 3 0 299
./bin/D-GPUGST 2 ../../../data/ Twitch 5 4 0 299
./bin/D-GPUGST 2 ../../../data/ Twitch 7 4 0 299



 cd ../../../code/GPUGST+
mkdir build
cd build
cmake ..
make 
 ./bin/GPUGST+ 1 ../../../data/ Twitch 3 0 299
 ./bin/GPUGST+ 1 ../../../data/ Twitch 5 0 299
 ./bin/GPUGST+ 1 ../../../data/ Twitch 7 0 299

 cd ../../../code/D-GPUGST+
mkdir build
cd build
cmake ..
make 
./bin/D-GPUGST+ 1 ../../../data/ Twitch 3 4 0 299
./bin/D-GPUGST+ 1 ../../../data/ Twitch 5 4 0 299
./bin/D-GPUGST+ 1 ../../../data/ Twitch 5 2 0 299
./bin/D-GPUGST+ 1 ../../../data/ Twitch 5 3 0 299
./bin/D-GPUGST+ 1 ../../../data/ Twitch 7 4 0 299