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
 

cd ../../../code/TrimCDP
mkdir build
cd build
cmake ..
make 
 ./bin/TrimCDP 2 ../../../data/ Twitch 3 0 299
 ./bin/TrimCDP 2 ../../../data/ Twitch 5 0 299
  ./bin/TrimCDP 2 ../../../data/ Twitch 7 0 299


 cd ../../../code/D-TrimCDP
mkdir build
cd build
cmake ..
make 
 ./bin/D-TrimCDP 2 ../../../data/ Twitch 3 4 0 299
./bin/D-TrimCDP 2 ../../../data/ Twitch 5 2 0 299
./bin/D-TrimCDP 2 ../../../data/ Twitch 5 3 0 299
./bin/D-TrimCDP 2 ../../../data/ Twitch 5 4 0 299
./bin/D-TrimCDP 2 ../../../data/ Twitch 7 4 0 299



 cd ../../../code/TrimCDP-WB
mkdir build
cd build
cmake ..
make 
 ./bin/TrimCDP-WB 1 ../../../data/ Twitch 3 0 299
 ./bin/TrimCDP-WB 1 ../../../data/ Twitch 5 0 299
 ./bin/TrimCDP-WB 1 ../../../data/ Twitch 7 0 299

 cd ../../../code/D-TrimCDP-WB
mkdir build
cd build
cmake ..
make 
./bin/D-TrimCDP-WB 1 ../../../data/ Twitch 3 4 0 299
./bin/D-TrimCDP-WB 1 ../../../data/ Twitch 5 4 0 299
./bin/D-TrimCDP-WB 1 ../../../data/ Twitch 5 2 0 299
./bin/D-TrimCDP-WB 1 ../../../data/ Twitch 5 3 0 299
./bin/D-TrimCDP-WB 1 ../../../data/ Twitch 7 4 0 299