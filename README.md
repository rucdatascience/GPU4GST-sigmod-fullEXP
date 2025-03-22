# Optimal Group Steiner Tree Search on GPUs

## GST_data
The dataset of the paper is stored on [Onedrive](https://1drv.ms/f/c/683d9dd9f262486b/Ek6Fl_brQzhDnI2cmhGIHxMBQ-L1ApeSqxwZKE4NBsDXSQ?e=3RBc8S) for download. In the following code, the dataset is stored in the "data" folder by default. There are eight datasets: Twitch, Musae, Github,  Youtube, Orkut, DBLP, Reddit, LiveJournal. There are 5 files for each dataset. For example, the Twitch dataset contains the following 5 files:
1. "Twitch.in". This readable file contains the basic information of this dataset. The two numbers on the first line of the file represent the number of vertices and edges in the graph. The following lines have three numbers representing the two end vertices and the weight of an edge. For example, "18 14919 100" shows that there is an edge between vertex 18 and vertex 14919, with an edge weight of 100.

2. "Twitch_beg_pos.bin". This is a binary file. The original file has V elements, each element representing the starting position of a vertex's adjacency list. Therefore, the position of a vertex can be obtained by subtracting the starting position of the next vertex from the starting position of that vertex.

3. "Twitch_csr.bin". This is a binary file. The original file has E elements, and this file stores an adjacency list of vertices, where each element represents an endpoint of an edge.

4. "Twitch_weight.bin". This is a binary file. The original file has E elements, which store the weights of edges, with each element representing the weight of an edge.

5. "Twitch.g". Each line of this file represents which vertices in the graph are included in a group. For example, "g7:2705 13464 16088 16341 22323" indicates that group 7 contains five vertices: 2705, 13464, 16088, 16341, and 22323.

## Running code example
Here, we show how to build and run experiment on a Linux server with the Ubuntu 20.04 system, an Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz, and 1 NVIDIA GeForce RTX A6000 GPU. The environment is as follows:
- gcc version 9.3.0 (GCC)
- CUDA compiler NVIDIA 11.8.89
- cmake version 3.28.3
- Boost
We will provide a detailed introduction to the experimental process as follows.

In your appropriate directory, execute the following commands:

Download the code:
```
git clone https://anonymous.4open.science/r/GPU4GST/
```
Switch the working directory to TrimCDP.
```
cd TrimCDP
```
Download the dataset from [OneDrive](https://1drv.ms/f/c/683d9dd9f262486b/Ek6Fl_brQzhDnI2cmhGIHxMBQ-L1ApeSqxwZKE4NBsDXSQ?e=3RBc8S). Assume that the dataset is located in the "data" folder of the working directory TrimCDP.



After preparing the environment according to the above suggestions, we can use the sh files in the "sh" folder to compile and run the code.
Among them, example.sh conducts experiments on six algorithms using the Twitch dataset, with each algorithm executing 50 queries of size 3. The running instruction is:
 ```
sh sh/example.sh
 ```
The experiment results will be automatically saved as CSV files in the "data/result" folder.

Run a complete experiment of six algorithms on one datasets using run.sh. Execute 300 queries of sizes 3, 5, and 7. The running instruction is:

 ```
sh sh/exp_one_dataset.sh
 ```

The other six sh files correspond to complete experiments of an algorithm on eight datasets, with 300 queries of sizes 3, 5, and 7 executed on each dataset. For example, to run an experiment on TrimCDP-WB, using instruction:

 ```
sh sh/exp_TrimCDP-WB.sh
 ```
Taking example.sh as an example, explain the contents of the sh file as follows:
```
cd code/D-PrunedDP++
mkdir build
cd build
cmake ..
make
```
The above instructions switch to the corresponding directory of the algorithm and compile the code into an executable file.
```
./bin/cpudgst 2 ../../../ data/ Twitch 3 4 0 10
```
This instruction executes the executable file, specifying the query size, the dataset to be used and its location, the upper bound of the diameter constraint, and the start and end indices of the queries to be executed.
## GST_code
All code is located in the 'Code' folder. There are six subfolders, each corresponding to the code of one of the six experiments in section 5.1 of the paper.
- PrunedDP++. This is the PrunedDP++ version code of GST without diameter constraints.
- TrimCDP. This is the TrimCDP version code without diameter constraint for GST.
- TrimCDP-WB. This is the TrimCDP-WB version code without diameter constraint for GST.
- D-PrunedDP++. This is the PrunedDP++ version code with diameter constraints for GST.
- D-TrimCDP. This is the TrimCDP version code with diameter constraints for GST.
- D-TrimCDP-WB. This is the TrimCDP-WB version code with diameter constraints for GST.

In the six subfolders, there are .h, .cu, .cuh, and .cpp files used for conducting the experiments described in the paper. The .h and .cuh files are in the "include" directory, while the .cpp files are in the "src" directory. The following explanation uses code without diameter constraints as an example, and code with diameter constraints is similar.


### CPU:
- "PrunedDP++/src/main.cpp" contains the code for conducting the experiment in the paper. 
- "PrunedDP++/include/CPUNONHOP.h" contains the algorithm code for GST without diameter constraints.


### GPU
- "TrimCDP/src/main.cpp" contains the code for conducting the experiment in the paper. 
- "TrimCDP/include/exp_GPU_nonHop.h" contains code for reading in the graph, group, and queries.
- "TrimCDP/src/DPQ.cu" contains the algorithm code.


### GPU+
- "TrimCDP-WB/src/GSTnonHop.cu" contains the code for conducting the experiment in the paper. It also completes the tasks of reading in the graph, group, and queries.
- "TrimCDP-WB/include/mapper_enactor.cuh" contains the overall framework of the algorithm.
- "TrimCDP-WB/include/mapper.cuh" contains the code for performing specific operations on vertices, such as grow and merge operations.
- "TrimCDP-WB/include/reducer.cuh" contains the code for organizing and allocating work after completing vertices operations.

