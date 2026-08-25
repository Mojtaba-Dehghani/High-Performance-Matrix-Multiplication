# 🚀 High-Performance Matrix Multiplication Optimization Engine
### *From Naive O(N^3) Implementation to Hardware-Saturating Multi-Core SIMD Kernel Achieving a **3,176.44× Speedup***

[![Language](https://img.shields.io/badge/Language-C%20%2F%20C%2B%2B-00599C?logo=c&logoColor=white)](https://github.com/)
[![Architecture](https://img.shields.io/badge/Architecture-x86__64%20%7C%20AVX2%20%7C%20FMA-red?logo=intel&logoColor=white)](https://github.com/)
[![Parallelism](https://img.shields.io/badge/Parallelism-OpenMP%20%7C%20SIMD%20%7C%20ILP-orange)](https://github.com/)
[![Speedup](https://img.shields.io/badge/Speedup-3176.44x%20Cumulative-success?style=for-the-badge&logo=speedtest)](https://github.com/)
[![Academic](https://img.shields.io/badge/Course-Microprocessor%20Systems%20Design-blueviolet)](https://github.com/)

---

## 📌 Executive Summary

This repository documents the end-to-end performance engineering journey of optimizing dense matrix multiplication ($C = A \times B$) on modern multi-core x86-64 microprocessors. Starting from a classic naive triple-nested loop, each phase systematically exploits a distinct layer of hardware architecture:

1. **Phase 1: Low-Level Primitives & Cache Thrashing Analysis** (Pointers, Register Keywords, and $2^k$ Set-Associativity Stride Conflicts) $\rightarrow$ **$2.21\times$**
2. **Phase 2: Cache Hierarchy & Memory Subsystem Alignment** (Matrix Transposition vs. Hierarchical Tiling / Blocking) $\rightarrow$ **$24.07\times$**
3. **Phase 3: Instruction-Level Parallelism (ILP) & FPU Microarchitectural Probing** (Pipeline Depth Estimation & $6 \times 6$ Loop Unrolling) $\rightarrow$ **$2.75\times$**
4. **Phase 4: Multi-Level Cache-Friendly Blocking & 256-Bit SIMD Vectorization** (GCC `v4df` Vector Extensions & Broadcast Multiply-Accumulate) $\rightarrow$ **$1.65\times$**
5. **Phase 5: Multi-Core MIMD Parallelism** (OpenMP Multi-Threading with Scalable Core Allocation) $\rightarrow$ **$13.16\times$**
6. **Bonus Phase: AVX2 / FMA C-Intrinsics Micro-Kernel Optimization** (Hierarchical L1/L2 Blocking, Instruction Scheduling & Hardware Prefetch Tuning) $\rightarrow$ **$2.11\times$ over Baseline Vector Assembly**

$$\text{Total Cumulative Speedup} = 2.21 \times 24.07 \times 2.75 \times 1.65 \times 13.16 = \mathbf{3,176.44\times}$$

---

## 🖥️ Benchmark Hardware Environment

All experiments, microarchitectural probes, and benchmarks were executed on a bare-metal environment with the following specifications:

| Hardware Component | Technical Specification | Microarchitectural Relevance |
| :--- | :--- | :--- |
| **Processor** | **13th Gen Intel® Core™ i7-13650HX** | 14 Physical Cores (6 P-Cores + 8 E-Cores), 20 Threads |
| **Base Clock / Boost** | 2.60 GHz Base / up to 4.90 GHz Turbo | Out-of-Order Execution, Deep Execution Pipelines |
| **L1 Cache (Data + Inst)**| **1.2 MB Total** (Data: ~87.8 KB per core) | 4–5 cycle latency, private per core |
| **L2 Cache** | **11.5 MB Total** (~841 KB per core) | 14 cycle latency, non-inclusive / private per core |
| **L3 Cache (LLC)** | **24.0 MB Shared** | Shared across all cores, 50–70 cycle latency |
| **SIMD ISA Support** | AVX, AVX2, FMA3, SSE4.2 | 256-bit Vector Registers (`ymm0`–`ymm15`) |
| **Main Memory (RAM)** | DDR5 Dual-Channel High-Bandwidth | High latency (~60–80 ns), critical bottleneck for naive traversal |

---

## 📈 Optimization Progression & Cumulative Speedup

The chart and table below showcase the multiplicative performance gains across all 5 core phases:

```
[Phase 1] Baseline Naive Array Indexing  | 1.0x (Baseline)
[Phase 1] Pointers + Hardware Registers  | 2.21x
[Phase 2] Matrix Transposition           | 53.19x
[Phase 3] 6x6 ILP Loop Unrolling         | 146.29x
[Phase 4] SIMD 256-bit Vectorization     | 241.38x
[Phase 5] 14-Core OpenMP Multithreading  | 3,176.44x 🚀
```

### Overall Benchmark Summary Table ($N=2048$ to $N=8192$)

| Phase / Implementation Function | $N=2048$ (s) | $N=4000$ (s) | $N=4096$ (s) | $N=7000$ (s) | $N=8192$ (s) | Phase Speedup | Cumulative Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: `matrix_mult_index`** | 136.27 s | >1000 s | >1900 s | Timeout | >9300 s | **$1.00\times$** | **$1.00\times$** |
| **Phase 1: `matrix_mult_ptr_reg`** | 93.37 s | — | 1939.91 s | — | 9344.63 s | **$2.21\times$** | **$2.21\times$** |
| **Phase 2: `matrix_mult_transpose`** | 6.11 s | 51.35 s | 55.80 s | 284.66 s | 421.42 s | **$24.07\times$** | **$53.19\times$** |
| **Phase 3: `matrix_mult_unrolling`** | 2.63 s | 19.87 s | 20.23 s | 102.94 s | 162.49 s | **$2.75\times$** | **$146.29\times$** |
| **Phase 4: `..._cache_friendly_vec`**| 1.13 s | 7.89 s | 13.22 s | 42.43 s | 118.89 s | **$1.65\times$** | **$241.38\times$** |
| **Phase 5: `..._vec_omp` (14 Cores)**| **0.10 s** | **0.44 s** | **1.06 s** | **2.79 s** | **9.57 s** | **$13.16\times$** | **🚀 3,176.44×** |

---

## 🔬 Deep-Dive Architectural Breakdown

---

### 🔹 Phase 1: Micro-Optimizations & The $2^k$ Cache Stride Hazard
* **Pointers vs. Array Indexing:** Eliminates redundant address calculation arithmetic `((i * n) + j)` inside inner loops.
* **Register Allocation:** Guarantees hot accumulator variables are mapped to fast CPU general-purpose registers rather than spilling to the stack (`matrix_mult_ptr_reg` vs `matrix_mult_ptr_no_reg` yielded a **$2.14\times$** improvement).
* **The $N=1024$ Cache Thrashing Hazard:**
  * At $N=1024$ ($2^{10}$), memory addresses between successive rows map to identical cache set indices in set-associative L1/L2 caches.
  * When reading down matrix $B$ column-wise, every access evicts the previous cache line, causing catastrophic cache conflict misses and dropping performance below $N=1100$.

---

### 🔹 Phase 2: Cache Locality & Spatial Coherence
* **The Memory Stride Bottleneck:** Standard multiplication traverses matrix $B$ with stride $N \times 8$ bytes (column-major order), leading to nearly 100% L1/L2 cache misses on large matrices.
* **Matrix Transposition ($B^T$):** By pre-transposing $B$ into aligned memory via `_aligned_malloc(..., 64)`, column traversal becomes contiguous row traversal. Both $A$ and $B^T$ enjoy 100% spatial cache line reuse (64-byte cache line = 8 double-precision floats loaded per miss).
  * **Result:** Achieved **$24.07\times$** average speedup over Phase 1.
* **Hierarchical Tiling (Block Multiplication):**
  * Evaluated tile sizes: $B \in \{8, 16, 32, 64, 128, 256\}$.
  * *Crucial Finding:* The optimal block size shrunk from $64 \times 64$ at $N=2048$ to $16 \times 16$ at $N=8192$. As the full matrix size increases, smaller sub-blocks ensure that all 3 active sub-matrices ($A_{ik}, B_{kj}, C_{ij}$) fit strictly within private L1/L2 caches without eviction.

---

### 🔹 Phase 3: FPU Pipeline Probing & $6 \times 6$ ILP Unrolling
To determine the theoretical upper limit for loop unrolling without causing register pressure or execution stalls, specialized microbenchmarks probed the CPU's Floating-Point Units (FPUs):

* **FPU Micro-Benchmark Probe Results:**
  * **Adder Pipeline:** Saturated at 8 concurrent independent additions (<7% latency variation, spiking to +19% at 9 ops).
  * **Multiplier Pipeline:** Handled up to 10 pipelined multiply operations with zero penalty (<1% variation).
  * **Fused Multiply-Add (Simultaneous):** Maintained stable low latency up to 6 concurrent operations (+9% overhead at 6 ops, sharp knee to +29% at 7 ops and +45% at 8 ops).
* **$6 \times 6$ Outer Loop Unrolling Kernel:**
  * Unrolled both $i$ and $j$ loops by a factor of 6, calculating a $6 \times 6$ output tile (36 concurrent multiply-accumulates) per inner loop iteration.
  * Kept all 36 accumulator values in registers (`c00` to `c55`), fully saturating execution ports while avoiding register spills.
  * **Result:** Yielded an additional **$2.75\times$** speedup over transposition.

---

### 🔹 Phase 4: SIMD Vectorization with 256-Bit GCC Primitives
* **Scalar Bottleneck Removal:** Scalar code processes 1 double (64-bit) per cycle, leaving 75% of AVX execution units idle.
* **Vector Architecture (`v4df`):**
  * Utilized GCC vector extensions: `typedef double v4df __attribute__((vector_size(32)));`
  * Applied unaligned vector casting `uv4df` on memory loads to safely stream contiguous data.
  * Structured inner loop as a **$4 \times 8$ register micro-kernel** computing 8 vector registers (32 double-precision floats) per step using broadcast-multiply logic:
    $$c_{00} \mathrel{+}= a_0 \cdot b_{\text{vec}0}, \quad c_{01} \mathrel{+}= a_0 \cdot b_{\text{vec}1}, \quad \dots$$
  * Combined with $64 \times 64$ cache-friendly hierarchical blocking to eliminate L3 cache thrashing.
  * **Result:** Achieved an additional **$1.65\times$** speedup.

---

### 🔹 Phase 5: Multi-Core MIMD Parallelism (OpenMP)
* **Workload Distribution:** Parallelized outer block-distribution loops using `#pragma omp parallel for`.
* **Zero-Lock Architectural Design:**
  * Since each thread computes disjoint $64 \times 64$ sub-blocks of the output matrix $C$, no synchronizations, atomic operations, or critical sections were needed.
  * Loop index variables were privatized while maintaining hardware register mapping for inner accumulators (`c00`–`c31`).
  * Residual fringes (matrices non-divisible by block sizes) were cleanly partitioned across threads.
* **Parallel Scaling & Efficiency Analysis:**
  * Phase 5 speedup over Phase 4: **$13.16\times$**.
  * On a 14-core processor, achieving a **13.16× scaling factor (94% parallel efficiency)** proves optimal workload balance, negligible thread management overhead, and sufficient memory bandwidth to feed all 14 cores concurrently.

---

### 🌟 Bonus Phase: AVX2 / FMA C-Intrinsics & Micro-Kernel Tuning

Starting from a reference assembly micro-kernel (`matrix_mult_vector_4x32`), the following advanced optimizations were engineered:

1. **Assembly to C-Intrinsics Porting:** Refactored inline assembly (`__asm__`) to Intel AVX2/FMA intrinsics (`_mm256_fmadd_pd`, `_mm256_load_pd`, `_mm256_store_pd`, `_mm256_set1_pd`). This gave the compiler's instruction scheduler full visibility to optimize register spilling and out-of-order execution pipelines.
2. **Hierarchical Multi-Level Blocking:** Implemented $128 \times 128$ L1/L2 cache tiling.
3. **Register Micro-Kernel ($4 \times 8$):** Maintained 8 accumulator YMM vector registers, minimizing write-back traffic.
4. **Software Prefetch Pruning:** Empirically proved that removing explicit `__builtin_prefetch` instructions reduced pipeline stalls, allowing the hardware Stream Prefetcher to operate unimpeded.

#### 📊 Bonus Phase Benchmark Comparison

| Matrix Dimension ($N$) | Base Vector Assembly `4x32` | Optimized C-Intrinsics `4x32` | Speedup Factor |
| :---: | :---: | :---: | :---: |
| **$N = 1024$** | 0.095 s | **0.075 s** | **$1.27\times$** |
| **$N = 2048$** | 0.680 s | **0.555 s** | **$1.23\times$** |
| **$N = 4096$** | 5.895 s | **4.610 s** | **$1.28\times$** |
| **$N = 8192$** | 81.650 s | **36.640 s** | **$2.23\times$** |
| **Average Execution Time** | **22.08 s** | **10.47 s** | **🚀 2.11× Faster** |

---

## 🛠️ Build & Compilation Instructions

### Prerequisites
* GCC / G++ (>= 11.0 recommended) or Clang (>= 13.0)
* OpenMP 4.5+ support
* x86-64 CPU supporting AVX2 and FMA instructions

### Recommended Build Flags

```bash
# Compile with maximum optimizations, AVX2, FMA, and OpenMP support:
gcc -O3 -mavx2 -mfma -fopenmp -march=native -ffast-math -o matrix_engine main.c

# Run benchmark suite:
./matrix_engine
```

### In Code::Blocks IDE:
1. Navigate to **Project $\rightarrow$ Build Options $\rightarrow$ Compiler Flags**.
2. Enable:
   * `[-O3]` Optimization for maximum speed
   * `[-fopenmp]` OpenMP multithreading
   * `[-mavx2]` and `[-mfma]` instruction set architectures
3. In **Linker Settings**, append `-fopenmp` to Other Linker Options.

---

## 👥 Author & Academic Context

* **Author:** Mojtaba Dehghani Arani (مجتبی دهقانی آرانی)
* **Student ID:** 402101703
* **Course:** Microprocessor Systems Design (طراحی سیستم‌های میکروپروسسوری)
* **Supervising Professor:** Dr. Movahedin (دکتر موحدین)

---

<div align="center">
  <sub>Engineered with precision for high-performance computing, low-level architecture exploitation, and maximum hardware saturation.</sub>
</div>
