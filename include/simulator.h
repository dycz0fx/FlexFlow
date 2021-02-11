/* Copyright 2020 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _FLEXFLOW_SIMULATOR_H_
#define _FLEXFLOW_SIMULATOR_H_

#include "ffconst.h"
#include "config.h"
#include <memory>
#include <fstream>
#include <unordered_map>

class Conv2DMeta;
class LinearMeta;
class Pool2DMeta;
class ElementUnaryMeta;
class ElementBinaryMeta;
class SoftmaxMeta;
class BatchMatmulMeta;
//class BatchNormMeta;
class ConcatMeta;
//class DropoutMeta;
class TransposeMeta;
class Op;
class FFModel;

struct CostMetrics {
  float forward_time, backward_time;
  size_t memory_requirement;
};

class Device {
public:
    enum DeviceType {
        DEVICE_COMP,
        DEVICE_MEM,
        DEVICE_COMM,
    };
    Device(std::string name, DeviceType type, int node_id, int socket_id, int device_id);
    std::string name;
    DeviceType type;
    int node_id;
    int socket_id;
    int device_id;
};

class Comp_device : public Device {
public:
    enum CompDevType {
        LOC_PROC,   //CPU
        TOC_PROC,   //GPU
    };
    CompDevType comp_type;
    size_t capacity;
    Comp_device(std::string name, CompDevType comp_type, int node_id, int socket_id, int device_id);
};

class Mem_device : public Device {
public:
    enum MemDevType {
        SYSTEM_MEM,     // DRAM on a single node
        Z_COPY_MEM,     // Zero-copy memory betweeen CPU DRAM and all GPUs on a single node
        GPU_FB_MEM,     // GPU framebuffer memory for a single GPU
    };
    MemDevType mem_type;
    size_t capacity;
    Mem_device(std::string name, MemDevType mem_type, int node_id, int socket_id, int device_id, size_t capacity);
};

class Comm_device : public Device {
public:
    enum CommDevType {
        MEMBUS_COMM,
        UPI_IN_COMM,
        UPI_OUT_COMM,
        NIC_IN_COMM,
        NIC_OUT_COMM,
        PCI_TO_HOST_COMM,
        PCI_TO_DEV_COMM,
        NVLINK_COMM,
    };
    CommDevType comm_type;
    float latency;
    float bandwidth;
    Comm_device(std::string name, CommDevType comm_type, int node_id, int socket_id, int device_id, float latency, float bandwidth);
};

class MachineModel {
public:
  virtual int get_version() = 0;
  virtual Comp_device *get_gpu(int device_id) = 0;
  virtual Mem_device *get_gpu_fb_mem(int devicd_id) = 0;
  virtual int get_num_gpus() = 0;
  virtual float get_intra_node_gpu_bandwidth() = 0;
  virtual float get_inter_node_gpu_bandwidth() = 0;
  virtual std::vector<Comm_device *> get_comm_path(Mem_device *src_mem, Mem_device *tar_mem) = 0;
  virtual std::string to_string() = 0;
  int version;
};

class MachineModel_old : public MachineModel {
public:
  MachineModel_old(int num_nodes, int num_gpus_per_node, size_t capacity);
  int get_version();
  Comp_device *get_gpu(int device_id);
  Mem_device *get_gpu_fb_mem(int devicd_id);
  int get_num_gpus();
  float get_intra_node_gpu_bandwidth();
  float get_inter_node_gpu_bandwidth();
  std::vector<Comm_device *> get_comm_path(Mem_device *src_mem, Mem_device *tar_mem);
  std::string to_string();
private:
  int num_nodes;
  int num_gpus_per_node;
  int num_gpus;
  float inter_gpu_bandwidth;
  float inter_node_bandwidth;
  float gpu_dram_bandwidth;
  std::map<int, Comp_device*> id_to_gpu;
  std::map<int, Mem_device*> id_to_gpu_fb_mem;
  std::map<int, Comm_device*> id_to_gputodram_comm_device;
  std::map<int, Comm_device*> id_to_dramtogpu_comm_device;
  std::map<size_t, Comm_device*> ids_to_inter_gpu_comm_device;
  std::map<size_t, Comm_device*> ids_to_inter_node_comm_device;
};

class MachineModel_new : public MachineModel {
public:
    enum NicDistribution {
      PER_NODE,
      PER_SOCKET,
    };
    MachineModel_new(std::string file, size_t gpu_fb_mem_capacity);
    int get_version();
    Comp_device *get_cpu(int device_id);
    Comp_device *get_cpu(int socket_id, int local_id);
    Comp_device *get_gpu(int device_id);
    Comp_device *get_gpu(int socket_id, int local_id);
    Mem_device *get_sys_mem(int socket_id);
    Mem_device *get_z_copy_mem(int socket_id);
    Mem_device *get_gpu_fb_mem(int device_id);
    Mem_device *get_gpu_fb_mem(int socket_id, int local_id);
    Comm_device *get_nvlink(Mem_device *src_mem, Mem_device *tar_mem);
    int get_num_gpus();
    float get_intra_node_gpu_bandwidth();
    float get_inter_node_gpu_bandwidth();
    std::vector<Comm_device *> get_comm_path(Mem_device *src_mem, Mem_device *tar_mem);
    std::string to_string();
private:
    int num_nodes;
    int num_sockets_per_node;
    int num_cpus_per_socket;
    int num_gpus_per_socket;
    int num_sockets;
    int num_cpus;
    int num_gpus;
    int num_nvlinks_per_node;
    float membus_latency;
    float membus_bandwidth;
    float upi_latency;
    float upi_bandwidth;
    float nic_latency;
    float nic_bandwidth;
    NicDistribution nic_distribution;
    float pci_latency;
    float pci_bandwidth;
    float nvlink_latency;
    float nvlink_bandwidth;
    size_t gpu_fb_mem_capacity;
    std::vector<Comm_device::CommDevType> intra_socket_sys_mem_to_sys_mem;
    std::vector<Comm_device::CommDevType> inter_socket_sys_mem_to_sys_mem;
    std::vector<Comm_device::CommDevType> inter_node_sys_mem_to_sys_mem;
    std::vector<Comm_device::CommDevType> intra_socket_sys_mem_to_gpu_fb_mem;
    std::vector<Comm_device::CommDevType> inter_socket_sys_mem_to_gpu_fb_mem;
    std::vector<Comm_device::CommDevType> inter_node_sys_mem_to_gpu_fb_mem;
    std::vector<Comm_device::CommDevType> intra_socket_gpu_fb_mem_to_sys_mem;
    std::vector<Comm_device::CommDevType> inter_socket_gpu_fb_mem_to_sys_mem;
    std::vector<Comm_device::CommDevType> inter_node_gpu_fb_mem_to_sys_mem;
    std::vector<Comm_device::CommDevType> intra_socket_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<Comm_device::CommDevType> inter_socket_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<Comm_device::CommDevType> inter_node_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<std::vector<Comp_device *>> cpus;   // socket_id, local_id
    std::vector<std::vector<Comp_device *>> gpus;   // socket_id, local_id
    std::vector<Mem_device *> sys_mems;             // socket_id
    std::vector<Mem_device *> z_copy_mems;          // socket_id
    std::vector<std::vector<Mem_device *>> gpu_fb_mems;     // socket_id, local_id
    std::vector<Comm_device *> membuses;            // socket_id
    std::vector<Comm_device *> upi_ins;             // socket_id
    std::vector<Comm_device *> upi_outs;            // socket_id
    std::vector<Comm_device *> nic_ins;             // socket_id
    std::vector<Comm_device *> nic_outs;            // socket_id
    std::vector<Comm_device *> pcis_to_host;             // from gpu to main memory, socket_id
    std::vector<Comm_device *> pcis_to_device;            // from main memory to gpu, socket_id
    std::vector<std::vector<Comm_device *>> nvlinks;    // node_id, local_id
    std::unordered_map<size_t, Comm_device *> mem_to_nvlink;
    void set_comm_path(std::vector<Comm_device::CommDevType> &comm_path, std::string device_str);
    void add_cpus();
    void add_gpus();
    void add_membuses(float latency, float bandwidth);
    void add_upis(float latency, float bandwidth);
    void add_nics(float latency, float bandwidth, NicDistribution nic_distribution);
    void add_pcis(float latency, float bandwidth);
    void add_nvlinks(float latency, float bandwidth);
    void attach_nvlink(Mem_device *src_mem, Mem_device *tar_mem, Comm_device *comm);
    void add_comm_path(std::vector<Comm_device::CommDevType> comm_device_list, Mem_device *src_mem, Mem_device *tar_mem, std::vector<Comm_device *> &ret);
};

class SimTask {
public:
  enum SimTaskType {
    TASK_FORWARD,
    TASK_BACKWARD,
    TASK_COMM,
    TASK_UPDATE,
    TASK_BARRIER,
  };
  SimTask();
  void add_next_task(SimTask* task);
public:
  float ready_time, run_time;
  SimTaskType type;
  Device* device;
  Mem_device *mem;
  int counter;
  std::vector<SimTask*> next_tasks;
  std::string name;
  std::string get_type_str() const;
};

template <typename T>
class DotFile {
private:
  size_t node_id;
  std::map<T,size_t> node_ids;
  std::unique_ptr<std::ostream> out;
  std::string get_node_name(size_t node_id) const {
    std::ostringstream s;
    s << "node" << node_id;
    return s.str();
  }
public:
  DotFile() : node_id(0) {}
  DotFile(std::string const &filename) : DotFile(std::unique_ptr<std::ostream>(new std::ofstream(filename))) {}
  DotFile(std::unique_ptr<std::ostream> s)
    : node_id(0), out(std::move(s))
  {
    *out << "digraph taskgraph {";
  }

  void set_filename(std::string filename) {
    this->out = std::unique_ptr<std::ostream>(new std::ofstream(filename));
    *out << "digraph taskgraph {";
  }
  void reserve_node(T const &t) {
    if (this->node_ids.find(t) == this->node_ids.end()) {
      this->node_ids[t] = this->node_id++;
    }
  }
  void add_node(T const &t, std::map<std::string, std::string> const &params) {
    this->reserve_node(t);
    *out << "  " << this->get_node_name(this->node_ids.at(t)) << " [";
    for (auto it = params.begin(); it != params.end(); ++it)  {
      *out << it->first << "=" << it->second;
      if (std::next(it) != params.end()) {
        *out << ",";
      }
    }
    *out << "];" << std::endl;
  }
  void add_edge(T const &src, T const &dst) {
    this->reserve_node(src);
    this->reserve_node(dst);
    auto src_name = this->get_node_name(this->node_ids.at(src));
    auto dst_name = this->get_node_name(this->node_ids.at(dst));
    *out << "  " << src_name << " -> " << dst_name << ";" << std::endl;
  }
  void close() {
    *out << "}";
    out->flush();
  }
};

class SimTaskCompare {
public:
  bool operator() (SimTask* lhs, SimTask* rhs) {
    return lhs->ready_time > rhs->ready_time;
  }
};

class TaskManager {
public:
  TaskManager(size_t max_num_tasks);
  void reset();
  SimTask* new_barrier_task();
  SimTask* new_update_task();
  SimTask* new_comm_task();
  SimTask* new_comm_task(std::string name, Comm_device *comm_device, size_t message_size);
  SimTask* new_forward_task(Op* op, int idx);
  SimTask* new_backward_task(Op* op, int idx);
  SimTask* get_forward_task(Op* op, int idx);
  SimTask* get_backward_task(Op* op, int idx);
private:
  SimTask* new_task();
public:
  size_t global_task_id, max_num_tasks;
  SimTask** tasks;
  std::map<size_t, SimTask*> hash_to_forward_task, hash_to_backward_task;
};

class Simulator {
public:
  Simulator(const FFModel* model,
            FFHandler handler,
            Memory memory,
            MachineModel *machine);
  ~Simulator(void);
  void free_all();
  void* allocate(size_t num_elements, DataType type);
  void add_task_dependencies_with_xfer(
      SimTask* src_task, SimTask* dst_task, size_t message_size);
  CostMetrics measure_operator_cost(Op* op, const ParallelConfig& config);
  float simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode);
  float simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode,
      std::string const &export_file_name);
  static void strategy_search_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime);
public:
  Realm::RegionInstance simulatorInst;
  MachineModel *machine;
  Memory memory;
  FFHandler handler;
  char* base_ptr;
  size_t capacity;
  off_t offset;
  int warmup_times, repeat_times;
  TaskManager* task_manager;
  CompMode computationMode;
  cudaEvent_t start_event, end_event;
  std::map<size_t, CostMetrics> hash_to_operator_cost;
public:
  Conv2DMeta* conv2d_meta;
  LinearMeta* linear_meta;
  Pool2DMeta* pool2d_meta;
  ElementUnaryMeta* ele_unary_meta;
  ElementBinaryMeta* ele_binary_meta;
  SoftmaxMeta *softmax_meta;
  BatchMatmulMeta *batch_matmul_meta;
  //BatchNormMeta *batch_norm_meta;
  ConcatMeta *concat_meta;
  //DropoutMeta *dropout_meta;
  TransposeMeta *transpose_meta;
  int segment_size;
  int max_num_segments;
};
#endif
