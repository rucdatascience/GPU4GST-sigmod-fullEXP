#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_set>
#include <iostream>
#include <vector>
#include <boost/heap/fibonacci_heap.hpp> 

// Per user instruction, these are linked via cmake


#define inf 100000

struct records
{   int process_queue_num=0;
    int counts=0;
};

struct queue_element_d
{
    int v, p, d, cost;
    queue_element_d(int _v = 0, int _p = 0, int _d = 0, int _cost = 0)
        : v(_v), p(_p), d(_d), cost(_cost) {}
};

struct node
{
	short update = 0;
	short  type;
	std::atomic<int> cost;
	int u;
	short p1, p2, d1, d2;

    node() : update(0), type(0), u(0), p1(0), p2(0), d1(0), d2(0) {
        cost.store(inf);
    }
    node(const node& other)
        : update(other.update), type(other.type), u(other.u), p1(other.p1), p2(other.p2), d1(other.d1), d2(other.d2) {
        cost.store(other.cost.load());
    }
    node& operator=(const node& other) {
        if (this != &other) {
            update = other.update;
            type = other.type;
            cost.store(other.cost.load());
            u = other.u;
            p1 = other.p1;
            p2 = other.p2;
            d1 = other.d1;
            d2 = other.d2;
        }
        return *this;
    }
};

void atomic_fetch_min(std::atomic<int>& obj, int val) {
    int old_val = obj.load(std::memory_order_relaxed);
    while (old_val > val && !obj.compare_exchange_weak(old_val, val, std::memory_order_release, std::memory_order_relaxed));
}

class ThreadPool {
public:
  ThreadPool(size_t numThreads) : stop(false) {
    numThreads = std::min(numThreads, size_t(50));
    for (size_t i = 0; i < numThreads; ++i) {
      workers.emplace_back([this, i] {
        while (true) {
          std::function<void(size_t)> task;
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
          task(i);
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
  
  size_t get_thread_count() const { return workers.size(); }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void(size_t)>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

std::vector<std::vector<int>> non_overlapped_group_mt;

void set_max_ID_mt(graph_v_of_v_idealID &group_graph, std::unordered_set<int> &cumpulsory_group_vertices, std::vector<std::vector<std::vector<node>>>& host_tree, std::unordered_set<int> &contain_group_vertices)
{
    int bit_num = 1, v;
    for (auto it = cumpulsory_group_vertices.begin(); it != cumpulsory_group_vertices.end(); it++, bit_num <<= 1)
    {
        for (size_t to = 0; to < group_graph[*it].size(); to++)
        {
            v = group_graph[*it][to].first;
            host_tree[v][bit_num][0].cost.store(0);
            contain_group_vertices.insert(v);
        }
    }
}
int get_max_mt(int vertex, std::vector<std::vector<std::vector<node>>>& host_tree, int width)
{
    int re = 0;
    for (size_t i = 1; i < width; i <<= 1)
    {
        if (host_tree[vertex][i][0].cost.load(std::memory_order_relaxed) == 0)
        {
            re += i;
        }
    }
    return re;
}

void non_overlapped_group_init_mt(int group_sets_ID_range)
{
    std::vector<std::vector<int>>().swap(non_overlapped_group_mt);
    non_overlapped_group_mt.resize(group_sets_ID_range + 2);
    for (int i = 1; i <= group_sets_ID_range; i++)
    {
        for (int j = 1; j <= group_sets_ID_range; j++)
        { 
            if ((i & j) == 0)
            {
                non_overlapped_group_mt[i].push_back(j);
            }
        }
    }
}

graph_hash_of_mixed_weighted HOP_MT(std::unordered_set<int> &cumpulsory_group_vertices, graph_v_of_v_idealID &group_graph, graph_v_of_v_idealID &input_graph, int D, double &time_record, long long int &RAM, records &ret)
{
    int N = input_graph.size();
    int G = cumpulsory_group_vertices.size();

    int group_sets_ID_range = (1 << G) - 1;
    std::vector<std::vector<std::vector<node>>> host_tree(N);
    for (int i = 0; i < N; i++)
    {
        host_tree[i].resize(1 << G);
        for (int j = 0; j < (1 << G); j++)
        {
            host_tree[i][j].resize(D + 1);
        }
    }
    
    graph_hash_of_mixed_weighted solution_tree;
    std::atomic<int> min_cost(inf);
    non_overlapped_group_init_mt(group_sets_ID_range);

    std::unordered_set<int> contain_group_vertices;
    int width = 1 << G;
    std::vector<queue_element_d> current_level;
    set_max_ID_mt(group_graph, cumpulsory_group_vertices, host_tree, contain_group_vertices);

    for (int v = 0; v < N; v++)
    {
        int group_set_ID_v = get_max_mt(v, host_tree, width);
        for (int i = 1; i <= group_set_ID_v; i <<= 1)
        {
            if (i & group_set_ID_v)
            {
                host_tree[v][i][0].cost.store(0);
                host_tree[v][i][0].type = 0;
                current_level.push_back(queue_element_d(v, i, 0, 0));
            }
        }
    }

    auto begin = std::chrono::high_resolution_clock::now();
    int process = 0;
    std::atomic<int> f_v(-1), f_p(-1), f_d(-1);
    
    ThreadPool pool(32);
    size_t num_threads = pool.get_thread_count();
    if (num_threads == 0) num_threads = 1;
    
    std::vector<std::vector<queue_element_d>> per_thread_next_queues(num_threads);

    std::mutex main_loop_mutex;
    std::condition_variable cv_main_loop;
    int max_queue_size = 0;
    while (!current_level.empty())
    {
        process += current_level.size();
        max_queue_size = std::max(max_queue_size, process);
        for(auto& q : per_thread_next_queues) {
            q.clear();
        }

        size_t total_nodes = current_level.size();
        size_t chunk_size = (total_nodes + num_threads - 1) / num_threads;
        size_t num_chunks = (total_nodes > 0) ? (total_nodes + chunk_size - 1) / chunk_size : 0;
        
        if (num_chunks == 0) {
            continue;
        }

        std::atomic<size_t> chunks_finished = {0};

        for (size_t i = 0; i < num_chunks; ++i) {
            pool.enqueue([&, i, chunk_size, total_nodes, num_chunks](size_t thread_id) {
                size_t start = i * chunk_size;
                size_t end = std::min(start + chunk_size, total_nodes);

                for (size_t j = start; j < end; ++j) {
                    const auto& top_node = current_level[j];
                    int v = top_node.v;
                    int p = top_node.p;
                    int cost = top_node.cost;
                    int d = top_node.d;
                    int local_min_cost = min_cost.load(std::memory_order_relaxed);

                    if (cost != host_tree[v][p][d].cost.load(std::memory_order_relaxed)) {
                        continue;
                    }

                    if (p == group_sets_ID_range)
                    {
                        if (cost < local_min_cost)
                        {
                            atomic_fetch_min(min_cost, cost);
                            f_v.store(v); f_p.store(p); f_d.store(d);
                        }
                        continue;
                    }
                    
                    if (cost >= local_min_cost) {
                        continue;
                    }

                    // grow
                    if (d < D)
                    {
                        for (const auto& edge : input_graph[v])
                        {
                            int u = edge.first;
                            int len = edge.second;
                            int new_cost = cost + len;
                            if (new_cost < host_tree[u][p][d + 1].cost.load(std::memory_order_relaxed))
                            {
                                atomic_fetch_min(host_tree[u][p][d + 1].cost, new_cost);
                                host_tree[u][p][d + 1].type = 1;
                                host_tree[u][p][d + 1].u = v;
                                per_thread_next_queues[thread_id].emplace_back(u, p, d + 1, new_cost);
                            }
                        }
                    }
                    // merge
                    int p1 = p, d1 = d;
                    for (auto p2 : non_overlapped_group_mt[p1])
                    {
                        for (int d2 = 0; d2 <= D - d1; d2++)
                        {
                            if(host_tree[v][p2][d2].cost.load(std::memory_order_relaxed) == inf) continue;

                            int p1_cup_p2 = p1 | p2;
                            int new_d = std::max(d1, d2);
                            int merge_tree_cost = cost + host_tree[v][p2][d2].cost.load(std::memory_order_relaxed);
                            
                            if (merge_tree_cost < host_tree[v][p1_cup_p2][new_d].cost.load(std::memory_order_relaxed))
                            {
                                atomic_fetch_min(host_tree[v][p1_cup_p2][new_d].cost, merge_tree_cost);
                                host_tree[v][p1_cup_p2][new_d].type = 2;
                                host_tree[v][p1_cup_p2][new_d].p1 = p1;
                                host_tree[v][p1_cup_p2][new_d].p2 = p2;
                                host_tree[v][p1_cup_p2][new_d].d1 = d1;
                                host_tree[v][p1_cup_p2][new_d].d2 = d2;
                                per_thread_next_queues[thread_id].emplace_back(v, p1_cup_p2, new_d, merge_tree_cost);
                            }
                        }
                    }
                }

                if (chunks_finished.fetch_add(1, std::memory_order_acq_rel) + 1 == num_chunks) {
                    std::unique_lock<std::mutex> lock(main_loop_mutex);
                    cv_main_loop.notify_one();
                }
            });
        }
        
        std::unique_lock<std::mutex> lock(main_loop_mutex);
        cv_main_loop.wait(lock, [&]{ return chunks_finished.load() >= num_chunks; });
        
        current_level.clear();
        size_t total_next_size = 0;
        for(const auto& q : per_thread_next_queues) {
            total_next_size += q.size();
        }
        current_level.reserve(total_next_size);
        for(auto& q : per_thread_next_queues) {
            current_level.insert(current_level.end(), std::make_move_iterator(q.begin()), std::make_move_iterator(q.end()));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    time_record = runningtime;

    begin = std::chrono::high_resolution_clock::now();
    std::queue<queue_element_d> Q;
    if (f_v.load() != -1) {
        Q.push(queue_element_d(f_v.load(), f_p.load(), f_d.load()));
    }

    while (Q.size())
    {
        queue_element_d statu = Q.front();
        Q.pop();
        int v = statu.v;
        int p = statu.p;
        int d = statu.d;
        graph_hash_of_mixed_weighted_add_vertex(solution_tree, v, 0);
        if (host_tree[v][p][d].type == 1)
        {
            int u = host_tree[v][p][d].u;
            int c_uv = graph_v_of_v_idealID_edge_weight(input_graph, u, v);
            graph_hash_of_mixed_weighted_add_edge(solution_tree, u, v, c_uv);
            Q.push(queue_element_d(u, p, d - 1));
        }
        if (host_tree[v][p][d].type == 2)
        {
            int p1 = host_tree[v][p][d].p1;
            int p2 = host_tree[v][p][d].p2;
            int d1 = host_tree[v][p][d].d1;
            int d2 = host_tree[v][p][d].d2;
            Q.push(queue_element_d(v, p1, d1));
            Q.push(queue_element_d(v, p2, d2));
        }
    }

    end = std::chrono::high_resolution_clock::now();
    runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
     long long int counts = 0;
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            for (size_t d = 0; d <= D; d++)
            {
                if (host_tree[i][j][d].cost.load()!=inf)
                {
                    counts++;
                }
                
            }
            
        }
        
    }
       ret.counts =  counts;
       ret.process_queue_num = process;
    cout<<"queue_size "<<max_queue_size<<"N*width*D "<<N*width*D<<"count "<<counts<<endl;
    long long int nl=N,wl=width,dl=D ;
    RAM = nl*wl*dl;
    RAM += ((counts+max_queue_size));
    return solution_tree;
} 