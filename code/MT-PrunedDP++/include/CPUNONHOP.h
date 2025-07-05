#pragma once

/*
Li, Rong-Hua, Lu Qin, Jeffrey Xu Yu, and Rui Mao. "Efficient and progressive
group steiner tree search." In Proceedings of the 2016 International Conference
on Management of Data, pp. 91-106. 2016.
*/
#include <iostream>
using namespace std;
#include <atomic>
#include <boost/heap/fibonacci_heap.hpp>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "graph_hash_of_mixed_weighted_MST_postprocessing.h"
#include "graph_v_of_v_idealID/common_algorithms/graph_v_of_v_idealID_shortest_paths.h"

const int inf = 100000;
#pragma region
struct records {
  long long int process_queue_num=0;
  long long int counts=0;
};
pair<vector<int>, vector<int>>
graph_v_of_v_idealID_PrunedDPPlusPlus_find_SPs_to_g(
    graph_v_of_v_idealID &group_graph, graph_v_of_v_idealID &input_graph,
    int g_vertex) {

  /*time complexity: O(|E|+|V|log|V|)*/

  int N = input_graph.size();

  /*add dummy vertex and edges; time complexity: O(|V|)*/
  input_graph.resize(
      N + 1); // add a dummy vertex N, which is considered as g_vertex
  auto ite = group_graph[g_vertex].end();
  for (auto it = group_graph[g_vertex].begin(); it != ite; it++) {

    graph_v_of_v_idealID_add_edge(input_graph, N, it->first,
                                  0); // add dummy edge
  }

  /*time complexity: O(|E|+|V|log|V|)*/
  vector<int> distances; // total vertex and edge weights of paths
  vector<int> predecessors;
  graph_v_of_v_idealID_shortest_paths(input_graph, N, distances, predecessors);

  for (auto it = group_graph[g_vertex].begin(); it != ite; it++) {
    graph_hash_of_mixed_weighted_binary_operations_erase(input_graph[it->first],
                                                         N);
  }
  input_graph.resize(N); // remove dummy vertex N

  distances.resize(N);    // remove dummy vertex N
  predecessors.resize(N); // remove dummy vertex N
  for (int i = 0; i < N; i++) {
    if (predecessors[i] == N) {
      predecessors[i] = i; // since N is not in predecessors, predecessors[i]
                           // points to i, i.e., the path ends at i
    }
  }
  for (size_t i = 0; i < distances.size(); i++) {
    if (distances[i] < 0) {
      // cout << "distance0 " << i << endl;
      exit(1);
    }
  }

  return {predecessors, distances};
}

std::unordered_map<int, pair<vector<int>, vector<int>>>
graph_v_of_v_idealID_PrunedDPPlusPlus_find_SPs(
    graph_v_of_v_idealID &input_graph, graph_v_of_v_idealID &group_graph,
    std::unordered_set<int> &cumpulsory_group_vertices) {

  /*time complexity: O(|T||E|+|T||V|log|V|)*/
  std::unordered_map<int, pair<vector<int>, vector<int>>> SPs_to_groups;
  for (auto it = cumpulsory_group_vertices.begin();
       it != cumpulsory_group_vertices.end(); it++) {
    SPs_to_groups[*it] = graph_v_of_v_idealID_PrunedDPPlusPlus_find_SPs_to_g(
        group_graph, input_graph, *it);
  }

  return SPs_to_groups;
}

#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus_find_SPs

#pragma region
struct graph_v_of_v_idealID_PrunedDPPlusPlus_min_node {
  int v;
  int p; // group_set_ID
  int priority_value;
  graph_v_of_v_idealID_PrunedDPPlusPlus_min_node(int vertex, int group_id,
                                                 int priority)
      : v(vertex), p(group_id), priority_value(priority) {}
};
struct tree_node {
  int v, p;
};
bool operator<(graph_v_of_v_idealID_PrunedDPPlusPlus_min_node const &x,
               graph_v_of_v_idealID_PrunedDPPlusPlus_min_node const &y) {
  return x.priority_value >
         y.priority_value; // < is the max-heap; > is the mean heap;
                           // PriorityQueue is expected to be a max-heap of
                           // integer values
}
typedef typename boost::heap::fibonacci_heap<
    graph_v_of_v_idealID_PrunedDPPlusPlus_min_node>::handle_type
    handle_graph_v_of_v_idealID_PrunedDPPlusPlus_min_node;
#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus priority queue

#pragma region
class graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node {
  /*this is like the tree T(v,p) in the DPBF paper*/

public:
  int type; // =0: this is the single vertex v; =1: this tree is built by grown;
            // =2: built by merge

  std::atomic<int> cost; // cost of this tree T(v,p);

  int u; // if this tree is built by grown, then it's built by growing edge
         // (v,u);

  int p1, p2; // if this tree is built by merge, then it's built by merge
              // T(v,p1) and T(v,p2);
  graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node()
      : type(0), u(0), p1(0), p2(0) {
        cost.store(inf);
      }

  graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node(const graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node& other)
      : type(other.type), u(other.u), p1(other.p1), p2(other.p2) {
      cost.store(other.cost.load());
  }
  
  graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node& operator=(const graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node& other) {
      if (this == &other) {
          return *this;
      }
      type = other.type;
      cost.store(other.cost.load());
      u = other.u;
      p1 = other.p1;
      p2 = other.p2;
      return *this;
  }
};
#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node

#pragma region
vector<vector<int>>
graph_v_of_v_idealID_PrunedDPPlusPlus_non_overlapped_group_sets(
    int group_sets_ID_range) {

  /*this function calculate the non-empty and non_overlapped_group_sets_IDs of
  each non-empty group_set ID;

  time complexity: O(4^|Gamma|), since group_sets_ID_range=2^|Gamma|;

  the original DPBF code use the same method in this function, and thus has the
  same O(4^|Gamma|) complexity;*/

  vector<vector<int>> non_overlapped_group_sets_IDs(
      group_sets_ID_range + 1); // <set_ID, non_overlapped_group_sets_IDs>

  for (int i = 1; i <= group_sets_ID_range;
       i++) { // i is a nonempty group_set ID
    non_overlapped_group_sets_IDs[i] = {};
    for (int j = 1; j < group_sets_ID_range;
         j++) {           // j is another nonempty group_set ID
      if ((i & j) == 0) { // i and j are non-overlapping group sets
        /* The & (bitwise AND) in C or C++ takes two numbers as operands and
        does AND on every bit of two numbers. The result of AND for each bit is
        1 only if both bits are 1.
        https://www.programiz.com/cpp-programming/bitwise-operators */
        non_overlapped_group_sets_IDs[i].push_back(j);
      }
    }
  }

  return non_overlapped_group_sets_IDs;
}
#pragma endregion                                                              \
    graph_v_of_v_idealID_PrunedDPPlusPlus_non_overlapped_group_sets

#pragma region
vector<vector<int>>
graph_v_of_v_idealID_PrunedDPPlusPlus_covered_uncovered_groups(
    int group_sets_ID_range,
    std::unordered_set<int> &cumpulsory_group_vertices) {

  /*time complexity: O(|Gamma|*2^|Gamma|); for each p \in
   * [1,group_sets_ID_range], this function calculate the groups that have not
   * been coverred by p*/

  vector<vector<int>> uncovered_groups(group_sets_ID_range + 1);

  for (int p = 1; p <= group_sets_ID_range; p++) {

    vector<int> unc_groups;

    int pow_num = 0;
    for (auto it = cumpulsory_group_vertices.begin();
         it != cumpulsory_group_vertices.end(); it++) {
      int id = pow(2, pow_num);
      if ((id | p) != p) {         // id is not covered by p
        unc_groups.push_back(*it); // *it is a group not covered by p
      }
      pow_num++;
    }
    uncovered_groups[p] = unc_groups;
  }

  return uncovered_groups;
}
#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus_covered_uncovered_groups

#pragma region
std::unordered_map<int, std::unordered_map<int, int>>
graph_v_of_v_idealID_PrunedDPPlusPlus_virtual_node_distances(
    graph_v_of_v_idealID &group_graph,
    std::unordered_set<int> &cumpulsory_group_vertices,
    std::unordered_map<int, pair<vector<int>, vector<int>>> &SPs_to_groups) {

  std::unordered_map<int, std::unordered_map<int, int>> virtual_node_distances;

  for (auto it = cumpulsory_group_vertices.begin();
       it != cumpulsory_group_vertices.end(); it++) {
    int g1 = *it;
    auto xx = SPs_to_groups.find(g1);
    for (auto it2 = cumpulsory_group_vertices.begin();
         it2 != cumpulsory_group_vertices.end(); it2++) {
      int g2 = *it2;
      if (g1 <= g2) {
        int distance = inf;
        auto pointer_begin = group_graph[g2].begin(),
             pointer_end = group_graph[g2].end();
        for (auto it2 = pointer_begin; it2 != pointer_end; it2++) {
          int dis = xx->second.second[it2->first];
          if (dis < distance) {
            distance = dis;
          }
        }

        virtual_node_distances[g1][g2] = distance;
        virtual_node_distances[g2][g1] = distance;
      }
    }
  }

  return virtual_node_distances;
}
#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus_virtual_node_distances

#pragma region
int graph_v_of_v_idealID_PrunedDPPlusPlus_vertex_group_set_ID(
    int vertex, graph_v_of_v_idealID &group_graph,
    std::unordered_set<int> &cumpulsory_group_vertices) {

  /*time complexity: O(|Gamma|); this function returns the maximum group set ID
   * for a single vertex*/

  int ID = 0;
  int pow_num = 0;
  for (auto it = cumpulsory_group_vertices.begin();
       it != cumpulsory_group_vertices.end(); it++) {
    // cout<<" vid "<<vertex<<" "<<ID<<endl;
    if (graph_v_of_v_idealID_contain_edge(group_graph, *it,
                                          vertex)) { // vertex is in group *it
      ID = ID + pow(2, pow_num);
    }
    pow_num++;
  }
  return ID;
}
#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus_vertex_group_set_ID

#pragma region
int graph_hash_of_mixed_weighted_PrunedDPPlusPlus_edge_weighted_vertex_group_vertex_2_group_set_ID(
    int group_vertex, std::unordered_set<int> &cumpulsory_group_vertices) {

  /*time complexity: O(|Gamma|); this function returns the maximum group set ID
   * for a single vertex*/

  int pow_num = 0;
  for (auto it = cumpulsory_group_vertices.begin();
       it != cumpulsory_group_vertices.end(); it++) {
    if (*it == group_vertex) {
      return pow(2, pow_num);
    }
    pow_num++;
  }
  return 0;
}

#pragma endregion                                                              \
    graph_hash_of_mixed_weighted_PrunedDPPlusPlus_edge_weighted_vertex_group_vertex_2_group_set_ID

#pragma region
struct graph_v_of_v_idealID_AllPaths_min_node {
  int v_i, v_j;
  int Xslash;         // group_set_ID
  int priority_value; // W(v_i,v_j,X)
};
bool operator<(graph_v_of_v_idealID_AllPaths_min_node const &x,
               graph_v_of_v_idealID_AllPaths_min_node const &y) {
  return x.priority_value >
         y.priority_value; // < is the max-heap; > is the min heap;
                           // PriorityQueue is expected to be a max-heap of
                           // integer values
}
typedef typename boost::heap::fibonacci_heap<
    graph_v_of_v_idealID_AllPaths_min_node>::handle_type
    handle_graph_v_of_v_idealID_AllPaths_min_node;

std::unordered_map<string, int> graph_v_of_v_idealID_AllPaths(
    std::unordered_set<int> &cumpulsory_group_vertices,
    std::unordered_map<int, std::unordered_map<int, int>>
        &virtual_node_distances) {

  std::unordered_map<string, int>
      W; // String is ("v_i" + "_" + "v_j" + "_" + "group set ID")

  boost::heap::fibonacci_heap<graph_v_of_v_idealID_AllPaths_min_node>
      Q; // min queue
  std::unordered_map<string, int> Q_priorities;
  std::unordered_map<string, handle_graph_v_of_v_idealID_AllPaths_min_node>
      Q_handles; // key is String is ("v_i" + "_" + "v_j" + "_" + "group set
                 // ID")

  /*D records the popped out optimal subtrees; String is ("v_i" + "_" + "v_j" +
   * "_" + "group set ID") */
  std::unordered_set<string> D;

  for (auto it = cumpulsory_group_vertices.begin();
       it != cumpulsory_group_vertices.end(); it++) {
    int p = *it;
    // int Xslash =
    // graph_hash_of_mixed_weighted_PrunedDPPlusPlus_edge_weighted_vertex_group_vertex_2_group_set_ID(p,
    // cumpulsory_group_vertices);
    graph_v_of_v_idealID_AllPaths_min_node x;

    // x.v_i = p;
    // x.v_j = p;
    // x.Xslash = Xslash;
    // x.priority_value = 0;
    // string handle_ID = to_string(p) + "_" + to_string(p) + "_" +
    // to_string(Xslash); Q_handles[handle_ID] = Q.push(x);
    // Q_priorities[handle_ID] = 0;

    /*the following code for Xslash=0 is not in 2016 paper, but is necessary for
     * computing every W(v_i,v_j,X)*/
    x.v_i = p;
    x.v_j = p;
    x.Xslash = 0;
    x.priority_value = 0;
    string handle_ID = to_string(p) + "_" + to_string(p) + "_" + to_string(0);
    Q_handles[handle_ID] = Q.push(x);
    Q_priorities[handle_ID] = 0;
  }

  while (Q.size() > 0) {

    graph_v_of_v_idealID_AllPaths_min_node top_node = Q.top();
    int v_i = top_node.v_i, v_j = top_node.v_j, Xslash = top_node.Xslash;
    int cost = top_node.priority_value;
    Q.pop();

    string handle_ID =
        to_string(v_i) + "_" + to_string(v_j) + "_" + to_string(Xslash);
    W[handle_ID] = cost;
    D.insert(handle_ID);
    Q_handles.erase(handle_ID);
    Q_priorities.erase(handle_ID);
    // cout << "Q pop " + handle_ID << " priority " << cost << endl;

    /*the following code is not in 2016 paper, but is necessary for computing
     * every W(v_i,v_j,X)*/
    for (int i = 0; i < Xslash; i++) {
      if ((i | Xslash) == Xslash) {
        handle_ID = to_string(v_i) + "_" + to_string(v_j) + "_" + to_string(i);
        if (W.count(handle_ID) == 0) {
          W[handle_ID] = cost;
        } else {
          if (W[handle_ID] > cost) {
            W[handle_ID] = cost;
          }
        }
      }
    }

    for (auto it = cumpulsory_group_vertices.begin();
         it != cumpulsory_group_vertices.end(); it++) {
      int p = *it;
      int p_setID =
          graph_hash_of_mixed_weighted_PrunedDPPlusPlus_edge_weighted_vertex_group_vertex_2_group_set_ID(
              p, cumpulsory_group_vertices);

      if ((p_setID | Xslash) != Xslash) { // p_setID is not covered by Xslash

        int Xwave = Xslash + p_setID;

        int cost_wave = cost + virtual_node_distances[v_j][p];
        handle_ID =
            to_string(v_i) + "_" + to_string(p) + "_" + to_string(Xwave);

        if (D.count(handle_ID) > 0) {
          continue;
        }

        graph_v_of_v_idealID_AllPaths_min_node x;
        x.v_i = v_i;
        x.v_j = p;
        x.Xslash = Xwave;
        x.priority_value = cost_wave;

        if (Q_handles.count(handle_ID) == 0) {
          Q_handles[handle_ID] = Q.push(x);
          Q_priorities[handle_ID] = cost_wave;
          // cout << "Q push " + handle_ID << " priority " << cost_wave << endl;
        } else {
          if (cost_wave < Q_priorities[handle_ID]) {
            Q.update(Q_handles[handle_ID], x); // O(1) for decrease key
            Q_priorities[handle_ID] = cost_wave;
            // cout << "Q update " + handle_ID << " priority " << cost_wave <<
            // endl;
          }
        }
      }
    }
  }

  // cout << "W.size(): " << W.size() << endl;

  return W;
}
#pragma endregion graph_v_of_v_idealID_AllPaths

#pragma region

#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus_build_tree

#pragma region
int graph_v_of_v_idealID_PrunedDPPlusPlus_LB_procedure(
    int v, int X, int cost, int group_sets_ID_range,
    vector<vector<int>> &uncovered_groups,
    std::unordered_map<int, pair<vector<int>, vector<int>>> &SPs_to_groups,
    std::unordered_map<string, int> &W, std::unordered_map<string, int> &W2) {

  int lb =
      0; // lb should be lower bound cost of a feasible solution contains T(v,X)

  if (group_sets_ID_range != X) {
    int X_slash =
        group_sets_ID_range - X; // X_slash \cup X equals group_sets_ID_range

    int lb1 = inf, lb2 = -1, lb_one_label = -1;
    vector<int> *pointer_1 = &(uncovered_groups[X]);
    for (auto it = (*pointer_1).begin(); it != (*pointer_1).end(); it++) {
      int i = *it; // from group i to node v
      int dis_v_i = SPs_to_groups[i].second[v];
      int xxx = inf;
      for (auto it2 = (*pointer_1).begin(); it2 != (*pointer_1).end(); it2++) {
        int j = *it2;
        int dis_v_j = SPs_to_groups[j].second[v];
        if (xxx > dis_v_j) {
          xxx = dis_v_j;
        }
        int lb1_value =
            (dis_v_i +
             W[to_string(i) + "_" + to_string(j) + "_" + to_string(X_slash)] +
             dis_v_j) /
            2;
        if (lb1 > lb1_value) {
          lb1 = lb1_value;
        }
      }

      int lb2_value =
          (dis_v_i + W2[to_string(i) + "_" + to_string(X_slash)] + xxx) / 2;
      if (lb2 < lb2_value) {
        lb2 = lb2_value;
      }
      if (lb_one_label < dis_v_i) {
        lb_one_label = dis_v_i;
      }
    }

    if (lb1 < lb2) {
      lb = lb2;
    } else {
      lb = lb1;
    }
    if (lb < lb_one_label) {
      lb = lb_one_label;
    }
    // printf("1=%d 2=%d 3=%d\n",lb1,lb2,lb_one_label);
    // cout << "lb_one_label=" << lb_one_label << endl;
  }

  return cost + lb;
}
#pragma endregion graph_v_of_v_idealID_PrunedDPPlusPlus_LB_procedure

void atomic_fetch_min(std::atomic<int>& obj, int val) {
    int old_val = obj.load(std::memory_order_relaxed);
    while (old_val > val && !obj.compare_exchange_weak(old_val, val, std::memory_order_release, std::memory_order_relaxed));
}

class ThreadPool {
public:
  ThreadPool(size_t numThreads) : stop(false) {
    // 限制最大线程数为8
    numThreads = std::min(numThreads, size_t(32));
    for (size_t i = 0; i < numThreads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty()) {
              return;
            }
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <class F> void enqueue(F &&f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

int cpu_mt(graph_v_of_v_idealID &input_graph, graph_v_of_v_idealID &group_graph,
           std::unordered_set<int> &cumpulsory_group_vertices,
           int maximum_return_app_ratio, long long int &RAM_MB, double &time_record,
           records &ret) {
  int group_sets_ID_range =
      pow(2, cumpulsory_group_vertices.size()) -
      1; // the number of group sets: 2^|Gamma|, including empty set;   |Gamma|
         // should be smaller than 31 due to precision
  /*this function returns the first found feasible solution that has an
   * approximation ratio not larger than maximum_return_app_ratio*/
  int bit_num = 0,  N = input_graph.size();
  int error_safeguard = 2;
  // cost of empty tree is inf
  long long int process = 0;
  /*finding lowest-weighted paths from groups to vertices; time complexity:
   * O(|T||E|+|T||V|log|V|); return {g_ID, { distances, predecessors }} */
  std::unordered_map<int, pair<vector<int>, vector<int>>> SPs_to_groups =
      graph_v_of_v_idealID_PrunedDPPlusPlus_find_SPs(input_graph, group_graph,
                                                     cumpulsory_group_vertices);

  /*initialize Q*/
  std::vector<tree_node> cur_queue,
      nxt_queue(N * (group_sets_ID_range + 1));
  std::atomic<int> nxt_queue_size(0);
  std::atomic<int> best_cost(inf);

  vector<vector<int>> non_overlapped_group_sets_IDs =
      graph_v_of_v_idealID_PrunedDPPlusPlus_non_overlapped_group_sets(
          group_sets_ID_range); // time complexity: O(4^|Gamma|)
  auto ite = non_overlapped_group_sets_IDs.end();

  /*initialize uncovered_groups;  <p, <uncovered_groups>>;  time complexity:
   * O(|Gamma|*2^|Gamma|)*/
  vector<vector<int>> uncovered_groups =
      graph_v_of_v_idealID_PrunedDPPlusPlus_covered_uncovered_groups(
          group_sets_ID_range, cumpulsory_group_vertices);
  auto ite1 = uncovered_groups.end();

  std::unordered_map<int, std::unordered_map<int, int>> virtual_distances =
      graph_v_of_v_idealID_PrunedDPPlusPlus_virtual_node_distances(
          group_graph, cumpulsory_group_vertices, SPs_to_groups);

  auto ite2 = virtual_distances.end();

  std::unordered_map<string, int> W = graph_v_of_v_idealID_AllPaths(
      cumpulsory_group_vertices,
      virtual_distances); // String is ("v_i" + "_" + "v_j" + "_" + "group set
                          // ID")
  std::unordered_map<string, int>
      W2; // String is ("v_i" + "_" + "group set ID")
  for (auto it = cumpulsory_group_vertices.begin();
       it != cumpulsory_group_vertices.end(); it++) {
    int v_i = *it;
    for (int Xslash = 1; Xslash <= group_sets_ID_range; Xslash++) {
      int dis = inf;
      for (auto it2 = cumpulsory_group_vertices.begin();
           it2 != cumpulsory_group_vertices.end(); it2++) {
        int v_j = *it2;
        string handle_ID =
            to_string(v_i) + "_" + to_string(v_j) + "_" + to_string(Xslash);
        if (dis > W[handle_ID]) {
          dis = W[handle_ID];
        }
      }
      string handle_ID = to_string(v_i) + "_" + to_string(Xslash);
      W2[handle_ID] = dis;
    }
  }

  /*initialize trees with vertices; time complexity: O(2^|Gamma|*|V|);
  every vertex v is associated with T(v,p) only when v covers p, otherwise the
  cost of T(v,p) is considered as inf;
  every vertex v is associated with at most 2^|Gamma| trees*/
  vector<std::vector<graph_v_of_v_idealID_PrunedDPPlusPlus_tree_node>> trees;
  trees.resize(N);
  for (int i = 0; i < N; i++) {
    trees[i].resize(group_sets_ID_range + 1);
  }
  for (int v = 0; v < N; v++) {
    int group_set_ID_v =
        graph_v_of_v_idealID_PrunedDPPlusPlus_vertex_group_set_ID(
            v, group_graph,
            cumpulsory_group_vertices); /*time complexity: O(|Gamma|)*/

    for (int p = 1; p <= group_set_ID_v;
         p++) { // p is non-empty; time complexity: O(2^|Gamma|)
      if ((p | group_set_ID_v) ==
          group_set_ID_v) { // p represents a non-empty group set inside
                            // group_set_ID_v, including group_set_ID_v

        /*T(v,p)*/
        trees[v][p].cost.store(0);
        trees[v][p].type = 0;

        /*insert T(v,p) into Q_T*/
        tree_node x;
        x.v = v;
        x.p = p;
        cur_queue.push_back(x);
        // cout << "initial Q push " + handle_ID << " priority: " <<
        // x.priority_value << endl;
      }
    }
  }

  /*D records the popped out optimal subtrees; String is "v_p" ("vertex_ID" +
   * "_" + "group set ID") */
  std::unordered_set<string> D;

  // cout << "group_sets_ID_range:" << group_sets_ID_range << endl;
  int Q_T_max_size = 0;

  /*Big while loop*/

  auto begin = std::chrono::high_resolution_clock::now();
  cout << "cur_queue.size() " << cur_queue.size() << endl;

  // 创建线程池，限制最大线程数为16
  cout<<"thread_num "<<std::thread::hardware_concurrency()<<endl;
  ThreadPool pool(std::min(std::thread::hardware_concurrency(), unsigned(32)));
  std::mutex main_loop_mutex;
  std::condition_variable cv_main_loop;
  int max_queue_size = 0;
  while (!cur_queue.empty()) {
    std::atomic<size_t> tasks_finished = {0};
    size_t total_tasks = cur_queue.size();
    nxt_queue.assign(N * (group_sets_ID_range + 1), {});
    nxt_queue_size = 0;
    max_queue_size = std::max(max_queue_size, int(cur_queue.size()));
    ret.process_queue_num += cur_queue.size();
    for(const auto& top_node : cur_queue){
        pool.enqueue([&]() {
          int v = top_node.v, X = top_node.p;
          int v_X_tree_cost = trees[v][X].cost.load(std::memory_order_relaxed);
          string handle_ID = to_string(v) + "_" + to_string(X);
          int local_best_cost = best_cost.load(std::memory_order_relaxed);

         

          // 构建可行解
          int feasible_solu_cost = v_X_tree_cost;
          for (auto it = uncovered_groups[X].begin();
               it != uncovered_groups[X].end(); it++) {
            if (SPs_to_groups[*it].second[v] == inf) {
              feasible_solu_cost = inf;
              break;
            } else {
              feasible_solu_cost += SPs_to_groups[*it].second[v];
            }
          }

          if (feasible_solu_cost < local_best_cost) {
            atomic_fetch_min(best_cost, feasible_solu_cost);
          }

          // 处理合并操作
          int X_slash = group_sets_ID_range - X;
          if (trees[v][X_slash].cost.load(std::memory_order_relaxed) != inf) {
            int merged_tree_cost = v_X_tree_cost + trees[v][X_slash].cost.load(std::memory_order_relaxed);
            int cost_Tvp1_cup_p2 = trees[v][group_sets_ID_range].cost.load(std::memory_order_relaxed);

            if (merged_tree_cost < cost_Tvp1_cup_p2) {
              atomic_fetch_min(trees[v][group_sets_ID_range].cost, merged_tree_cost);

              if (merged_tree_cost <= local_best_cost + error_safeguard) {
                if (merged_tree_cost < local_best_cost) {
                  atomic_fetch_min(best_cost, merged_tree_cost);
                }
              }
            }
          }

          // 处理生长操作
          if (v_X_tree_cost < local_best_cost / 2 + 2) {
            for (auto it = input_graph[v].begin(); it != input_graph[v].end();
                 it++) {
              int u = it->first, cost_euv = it->second;
              int grow_tree_cost = v_X_tree_cost + cost_euv;

              if (grow_tree_cost < trees[u][X].cost.load(std::memory_order_relaxed)) {
                atomic_fetch_min(trees[u][X].cost, grow_tree_cost);
                int lb = graph_v_of_v_idealID_PrunedDPPlusPlus_LB_procedure(
                    u, X, grow_tree_cost, group_sets_ID_range, uncovered_groups,
                    SPs_to_groups, W, W2);

                if (lb <= local_best_cost + error_safeguard) {
                  int index = nxt_queue_size.fetch_add(1);
                  nxt_queue[index] = {u, X};
                }
              }
            }

            // 处理合并操作
            for (auto it = non_overlapped_group_sets_IDs[X].begin();
                 it != non_overlapped_group_sets_IDs[X].end(); it++) {
              int p2 = *it;
              if (trees[v][p2].cost.load(std::memory_order_relaxed) != inf) {
                int p1_cup_p2 = X + p2;
                int merged_tree_cost = v_X_tree_cost + trees[v][p2].cost.load(std::memory_order_relaxed);

                if (merged_tree_cost < trees[v][p1_cup_p2].cost.load(std::memory_order_relaxed)) {
                  atomic_fetch_min(trees[v][p1_cup_p2].cost, merged_tree_cost);
                  trees[v][p1_cup_p2].type = 2;
                  trees[v][p1_cup_p2].p1 = X;
                  trees[v][p1_cup_p2].p2 = p2;

                  if (merged_tree_cost <= local_best_cost * 0.667 + 2) {
                    int lb = graph_v_of_v_idealID_PrunedDPPlusPlus_LB_procedure(
                        v, p1_cup_p2, merged_tree_cost, group_sets_ID_range,
                        uncovered_groups, SPs_to_groups, W, W2);

                    if (lb <= local_best_cost + 2) {
                      if (p1_cup_p2 == group_sets_ID_range &&
                          merged_tree_cost < local_best_cost) {
                        atomic_fetch_min(best_cost, merged_tree_cost);
                      }
                      int index = nxt_queue_size.fetch_add(1);
                      nxt_queue[index] = {v, p1_cup_p2};
                    }
                  }
                }
              }
            }
          }
          if (tasks_finished.fetch_add(1) + 1 == total_tasks) {
              std::unique_lock<std::mutex> lock(main_loop_mutex);
              cv_main_loop.notify_one();
          }
        });
    }

    std::unique_lock<std::mutex> lock(main_loop_mutex);
    cv_main_loop.wait(lock, [&]{ return tasks_finished.load() == total_tasks; });

    // 更新队列
    cur_queue.clear();
    if (nxt_queue_size > 0) {
        cur_queue.assign(nxt_queue.begin(), nxt_queue.begin() + nxt_queue_size);
    }
  }

  // 获取最终结果
  int final_best_cost = best_cost.load();
  for (int i = 0; i < N; i++) {
    final_best_cost = min(final_best_cost, trees[i][group_sets_ID_range].cost.load());
  }
    long long int counts = 0;
  for (size_t i = 0; i < N; i++)
  {
    for (size_t j = 0; j < group_sets_ID_range + 1; j++)
    {
      if (trees[i][j].cost.load()!=inf)
      {
        counts++;
      }
    }
  }
  ret.counts = counts;
  cout<<"queue_size "<<max_queue_size<<"N*width "<<N*(group_sets_ID_range+1)<<"count "<<counts<<endl;
  long long int nl=N,wl=group_sets_ID_range+1 ;
  RAM_MB = nl * wl;
  RAM_MB += ((counts+max_queue_size));
  auto end = std::chrono::high_resolution_clock::now();
  double runningtime =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9;
  time_record = runningtime;
  return final_best_cost;
}