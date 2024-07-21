# HPCG

## Purpose

HPCG is a benchmark which executes conjugate gradient iterations.
JSC intends to submit the procured system for inclusion in the HPCG benchmark list. The successful bidder is expected to support and facilitate this goal. The performance results of the HPCG benchmark are also used as part of the evaluation of the system offer.

## Source

A reference implementation for selected processors is available from the HPCG web page at

http://hpcg-benchmark.org

Candidates are free to use optimized, proprietary versions as long as these comply with the HPCG benchmark rules. A summary of currently known sources of selected HPCG variants (including version restrictions) is given in the following:

* CPU
    * x86_64
        * Download: https://github.com/hpcg-benchmark/hpcg
        * Version: >= 3.1, commitID e64982640f0aa83f851fe3e1405c61d9a6d7321c
        * License: https://github.com/hpcg-benchmark/hpcg/blob/master/LICENSE
    * ARM
        * Download: https://github.com/ARM-software/HPCG_for_Arm
        * Version: >= commitID 0eddbfdf85e0e0f8384150b6657e874356025319
        * License: https://github.com/ARM-software/HPCG_for_Arm/blob/master/LICENSE
* GPU
    * NVIDIA HPC-Benchmarks 21.4
        * Download: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks
        * Version: >= 21.4-hpcg container image providing HPCG-NVIDIA v1.0.0
        * License: Nvidia Developer account required
    * NVIDIA High Performance Conjugate Gradient Benchmark (HPCG)
        * Download: https://github.com/NVIDIA/nvidia-hpcg
        * Version: >= commitID 8d7f630195fafb10a23f525251c116ed602d0865
        * License: https://github.com/NVIDIA/nvidia-hpcg/blob/master/LICENSE
    * AMD rocHPCG (based on HPCG v3.1) source code
        * Download: https://github.com/ROCmSoftwarePlatform/rocHPCG
        * Version: >= 0.8.0, >= commitID 3a5e87eff6b2ce4c9ba01d273f93d4aff34c4653
        * License: https://github.com/ROCmSoftwarePlatform/rocHPCG/blob/develop/LICENSE.md

## Building

We recommend using the provided JUBE script, which handles the build process as well; see subsection _JUBE_ below. Only the Nvidia HPCG container image requires manual preprocessing.

HPCG can be built for CPU (x86_64 and Arm (Grace), HPCG reference implementation; Grace platform with nvidia-hpcg) as well as GPU (AMD: rocHPCG source code, Nvidia: HPCG container image and nvidia-hpcg). Best performance on the NVIDIA Grace-Hopper platform can be achieved with nvidia-hpcg. For optimal performance on ARM other than the NVIDIA Grace, however, it is highly recommended to use `HPCG_for_ARM` (see link above) instead, which was not integrated into the provided JUBE script.

### CPU (x86_64, Arm)

The HPCG reference implementation depends on very few other packages; it needs a recent compiler (e.g. GCC (g++) or Intel oneAPI DPC++/C++ Compiler (icpx)), MPI support, and CMake for installation. A script (`src/prepare_cpu.sh`) for downloading the source code is provided.

The source code can be built with:

```
cd src
sh prepare_cpu.sh
cd hpcg-cpu
mkdir build; cd build;
CXX=g++
cmake -DCMAKE_CXX_COMPILER=${CXX} -DHPCG_ENABLE_MPI=ON -DHPCG_ENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release .. && make -j 16
```

### GPU: Nvidia CUDA

This is only recommended as a sort of reference on CUDA <12.

After registration for NGC, the Nvidia HPC-Benchmarks 21.4 HPCG container image is available at

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks

On our existing HPC systems, it was noticed that this HPCG container does not support multinode setups via Apptainer; instead, only single-node jobs were successful. In the provided reference JUBE version, the binary is hence not executed within the context of the container. However, this limitation might not exist on other systems, so the candidate may decide freely whether to use the container or an extracted binary.

In order to omit the container, extract the `xhpcg` binary as well as the `hpcg.sh` proxy script from inside the container to the benchmark subfolder `src/hpcg-cuda/` on the host file system. The provided script `src/prepare_cuda.sh` copies the extracted `hpcg.sh` as `hpcg.bash` and therein updates the variable `XHPCG` with the absolute path to the extracted `xhpcg` binary.
The extracted binary depends on the CUDA toolkit (tested v11.5) and OpenMPI (tested v4.1.2); CUDA driver version >= 450.36 is required.

```
HPCG_CUDA="src/hpcg-cuda"
mkdir -p ${HPCG_CUDA}
# Apptainer example
  apptainer exec hpc-benchmarks\:21.4-hpcg.sif cp /workspace/hpcg-linux-x86_64/xhpcg ${HPCG_CUDA}/
  apptainer exec hpc-benchmarks\:21.4-hpcg.sif cp /workspace/hpcg-linux-x86_64/hpcg.sh ${HPCG_CUDA}/
# Docker example
  docker container cp <CONTAINER ID>:/workspace/hpcg-linux-x86_64/xhpcg ${HPCG_CUDA}/
  docker container cp <CONTAINER ID>:/workspace/hpcg-linux-x86_64/hpcg.sh ${HPCG_CUDA}/
#update a copy of hpcg.sh with new path to xhpcg and store it as ${HPCG_CUDA}/hpcg.bash
bash src/prepare_cuda.sh ${HPCG_CUDA}
```

### GPU, CPU (Grace), GPU+CPU: nvidia-hpcg

NVIDIA's Open Source version of their HPCG benchmark is available at

https://github.com/NVIDIA/nvidia-hpcg

A build script is provided, which requires several arguments and reads various environment variables:
```
    # env vars required by build process
    # for ParaStationMPI use $EBROOTPSMPI
    export MPI_PATH=$EBROOTOPENMPI
    export CUDA_PATH=$EBROOTCUDA
    export MATHLIBS_PATH=$$EBROOTNVHPC/Linux_x86_64/24.3/math_libs/12.3
    export NCCL_PATH=$$EBROOTNVHPC/Linux_x86_64/24.3/comm_libs/12.3/nccl
    # module NVPL is only available on JEDI
    export NVPL_SPARSE_PATH=$EBROOTNVPL
    # NVPL_PATH is required by Makefiles setup/Make.{CUDA_AARCH64,CUDA_X86}
    export NVPL_PATH=$NVPL_SPARSE_PATH
    # 1: CUDA on,  0: CUDA off
    # 1: GRACE on, 0: GRACE off
    # 1: NCCL on,  0: NCCL off
    # e.g. CUDA-only using NCCL
    export USE_CUDA=1
    export USE_GRACE=0 
    export USE_NCCL=1
    # list of arguments to build_sample.sh
    # $1 unused
    # $2 --> HPCG_COMMIT_HASH
    # $3 --> HPCG_VER_MAJOR
    # $4 --> HPCG_VER_MINOR
    # $5 --> USE_CUDA
    # $6 --> USE_GRACE
    # $7 --> USE_NCCL
    sh ./build_sample.sh 0 $HPCG_COMMIT_HASH $HPCG_VER_MAJOR $HPCG_VER_MINOR $USE_CUDA $USE_GRACE $USE_NCCL
```
When executing the binary you might have to preload `libcusparse.so` and/or `libnvpl_sparse.so` using LD_PRELOAD, because the benchmark requires a library version >12.1, which is checked at runtime.

### GPU: AMD ROCm

Using CMake, rocHPCG can be compiled for any supported AMD GPU architecture. It depends on CMake (3.10 or later), MPI, NUMA library, AMD ROCm platform (>=4.1, tested with 5.3), and rocPRIM. A script (`src/prepare_rocm.sh`) for downloading the source code is provided. The environment must provide the path to the ROCm installation via the variable `ROCM_PATH`.

```
cd src
sh prepare_rocm.sh
cd hpcg-rocm
mkdir -p build/release/reference; cd build/release/reference;
export ROCM_PATH=/opt/rocm; # default ROCm install path
cmake -DHPCG_OPENMP=true -DOPT_MEMMGMT=true -DOPT_DEFRAG=true -DGPU_AWARE_MPI=ON -DCMAKE_BUILD_TYPE=Release -DROCM_PATH=${ROCM_PATH} -DCMAKE_MODULE_PATH=${ROCM_PATH}/hip/lib/cmake/hip -DHPCG_REFERENCE=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON ../../../ && \
  make -j 16;
```

### Modification

It is allowed to introduce changes to the build process, if it is required to better adapt the code to the respective target platform. Note that any such changes must comply with the HPCG rules on optimization:

https://hpcg-benchmark.org/faq/index.html#71

### JUBE

The JUBE step `compile` takes care of building the benchmark. It either calls the download script (CPU, Nvidia opensource HPCG, rocHPCG) or modifies the path to the extracted `xhpcg` binary in `hpcg.sh` (Nvidia HPCG only), and it configures and builds the benchmarks in accordance with the outlined flags above; see the JUBE subsection of the _Execution_ section below.

In order to implement support for a new HPC system the file `stages_include.yml` has to be modified: i) Add a new entry for parameter *system*; ii) add a Python dictionary entry in parameter *modules* for the new system and list all supported parameters given below comment "combine stage, compiler and MPI" in `stages_include.yml`; iii) continue with modifying `benchmark/jube/hpcg.yml`, i.e., adding a new dictionary entry for your system in various parameters of parametersets **systemParameter** and **executeset**.

## Rules

The number of tasks per node and threads per task can be chosen freely as long as the configuration conforms to the official HPCG rules.

Furthermore, HPCG must use at least 80% of installed memory (CPU memory or GPU memory). If the CPU benchmark is executed on a CPU equipped with High Bandwidth Memory (HBM), then HPCG must be allocated within HBM using for example `numactl -m` and HPCG must use at least 80% of HBM.

To control the memory consumption of the benchmark, the local grid dimension must be adapted in the JUBE script (`benchmark/jube/hpcg.yml`; `hpcg_local_dim_{x,y,z}`). If JUBE is not used, the local grid dimension in the dat files provided in `src/dat-files/` are to be changed accordingly.

## Execution

We recommend using the provided JUBE script, which handles the execution of the benchmark; see the dedicated section below.

### Multi-Threading

HPCG can be run with different numbers of OpenMP threads per MPI task. They are to be specified with `OMP_NUM_THREADS`; in the JUBE file, the parameter `threadspertask` is used for this purpose.

### CPU (x86_64)

The executable of the HPCG benchmark is `xhpcg`, to be found in the build directory (e.g. `src/hpcg-cpu/build/xhpcg`) after the build process has completed successfully. The provided parameter file `src/dat-files/hpcg-cpu.dat` needs to be copied to the working directory.

```
cp ${BENCHMARK_BASEDIR}/src/dat-files/hpcg-cpu.dat /my/workdir/hpcg.dat
# execute the benchmark, e.g. 64 MPI tasks with 2 OMP threads each
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=2
cd /my/workdir
srun --cpu-bind=threads --nodes=1 --ntasks-per-node=64 --cpus-per-task=${OMP_NUM_THREADS} --threads-per-core=1 /my/exe/path/xhpcg
```

### Nvidia CUDA on x86_64 platform

#### Nvidia closed source HPC Benchmark

The executable of Nvidia's HPCG benchmark is `xhpcg`, to be found in the according container or extracted from it (see above). The provided parameter file `src/dat-files/hpcg-cuda.dat` needs to be given as an argument to the proxy script `src/hpcg-cuda/hpcg.bash`, which wraps `xhpcg` and maps local MPI ranks to a node's resources. Here is an example for JUWELS Booster:

```
# JUWELS booster
export OMP_NUM_THREADS=6
srun --threads-per-core=1 --nodes=1 --ntasks-per-node=4 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=none hpcg.bash --ucx-affinity mlx5_0:mlx5_1:mlx5_2:mlx5_3 --cpu-affinity 18-23:6-11:42-47:30-35 --mem-affinity 3:1:7:5 --gpu-affinity 0:1:2:3 --cpu-cores-per-rank ${OMP_NUM_THREADS} --dat hpcg-cuda.dat
```

#### Nvidia open source benchmark

* please note that Nvidia's script `src/hpcg-nvidia/bin/hpcg.sh` always sets `OMP_PROC_BIND=TRUE` and `OMP_PLACES=sockets`
* i.e. no matter what these env vars were set to before executing this Nvidia shell script, these env vars will always be overwritten with the above given values
```
# JUWELS booster
export OMP_NUM_THREADS=6
srun --threads-per-core=1 --nodes=1 --ntasks-per-node=4 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=none env LD_PRELOAD=$${EBROOTNVHPC}/Linux_x86_64/24.3/math_libs/12.3/lib64/libcusparse.so:$${EBROOTNVHPC}/Linux_x86_64/24.3/math_libs/12.3/lib64/libcublas.so:$${EBROOTNVHPC}/Linux_x86_64/24.3/math_libs/12.3/lib64/libcublasLt.so:$${EBROOTNVHPC}/Linux_x86_64/24.3/comm_libs/12.3/nccl/lib/libnccl.so sh hpcg-nvidia/bin/hpcg.sh --of 1 --ucx-affinity mlx5_0:mlx5_1:mlx5_2:mlx5_3 --cpu-affinity 18-23:6-11:42-47:30-35 --mem-affinity 3:1:7:5 --gpu-affinity 0:1:2:3 --dat hpcg.dat
```

### Nvidia CUDA/CPU on Grace-Hopper platform (Nvidia open source benchmark)

* please note that Nvidia's script `src/hpcg-nvidia/bin/hpcg-aarch64.sh` sets `OMP_PROC_BIND` and `OMP_PLACES` as follows:
```
export OMP_PROC_BIND=${OMP_PROC_BIND:-TRUE}
export OMP_PLACES=${OMP_PLACES:-sockets}
```
* i.e. any of these env vars, which has not been set before, will be exported by the shell script to the environment with the respective default value

#### CUDA

```
export OMP_NUM_THREADS=72
srun --threads-per-core=1 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=none --gpus-per-task=1 env LD_PRELOAD=$${EBROOTNVHPC}/Linux_aarch64/24.3/math_libs/12.3/lib64/libcusparse.so:$${EBROOTNVHPC}/Linux_aarch64/24.3/math_libs/12.3/lib64/libcublas.so:$${EBROOTNVHPC}/Linux_aarch64/24.3/math_libs/12.3/lib64/libcublasLt.so:$${EBROOTNVHPC}/Linux_aarch64/24.3/comm_libs/12.3/nccl/lib/libnccl.so sh hpcg-nvidia/bin/hpcg-aarch64.sh --of 1 --nx ${hpcg_local_dim_x} --ny ${hpcg_local_dim_y} --nz ${hpcg_local_dim_z} --rt ${hpcg_time} --cpu-affinity 0-71:72-143:144-215:216-287 --mem-affinity 0:1:2:3
```

#### CUDA+CPU

```
# do not set OMP_NUM_THREADS
srun --threads-per-core=1 --cpu-bind=none env LD_PRELOAD=$${EBROOTNVHPC}/Linux_aarch64/24.3/math_libs/12.3/lib64/libcusparse.so:$${EBROOTNVHPC}/Linux_aarch64/24.3/math_libs/12.3/lib64/libcublas.so:$${EBROOTNVHPC}/Linux_aarch64/24.3/math_libs/12.3/lib64/libcublasLt.so:$${EBROOTNVHPC}/Linux_aarch64/24.3/comm_libs/12.3/nccl/lib/libnccl.so:$${EBROOTNVPL}/lib/libnvpl_sparse.so sh hpcg-nvidia/bin/hpcg-aarch64.sh --of 1 --nx ${hpcg_local_dim_x} --ny ${hpcg_local_dim_y} --nz ${hpcg_local_dim_z} --rt ${hpcg_time} --exm 2 --ddm 2 --lpm 1 --g2c 64 --npx 4 --npy 2 --npz 1 --cpu-affinity 0-7:8-71:72-79:80-143:144-151:152-215:216-223:224-287 --mem-affinity 0:0:1:1:2:2:3:3
```

#### CPU

* we have noticed on JEDI that the Grace-only build of nvidia-hpcg crashes with the local grid size suggested by Nvidia in `src/hpcg-nvidia/bin/RUNNING-aarch64` (x=512,y=512,z=288)
* we therefore opted to use a smaller but stably running local grid size of x=512,y=256,z=288 until this issue has been resolved
```
export OMP_NUM_THREADS=72
srun --threads-per-core=1 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=none env LD_PRELOAD=${EBROOTNVPL}/lib/libnvpl_sparse.so sh hpcg-nvidia/bin/hpcg-aarch64.sh --of 1 --nx ${hpcg_local_dim_x} --ny ${hpcg_local_dim_y} --nz ${hpcg_local_dim_z} --rt ${hpcg_time} --exm 1 --cpu-affinity 0-71:72-143:144-215:216-287 --mem-affinity 0:1:2:3
```

### AMD ROCm

The executable of AMD's rocHPCG benchmark is `rochpcg`, to be found in the build directory (e.g. `src/hpcg-rocm/build/release/reference/bin`). The provided parameter file `src/dat-files/hpcg-rocm.dat` needs to be copied to the working directory.

```
cp ${BENCHMARK_BASEDIR}/src/dat-files/hpcg-rocm.dat /my/workdir/hpcg.dat
#execute benchmark, e.g. 8 MPI tasks with 4 OMP threads each:
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=4
cd /my/workdir
srun --cpu-bind=threads --nodes=1 --ntasks-per-node=8 --cpus-per-task=${OMP_NUM_THREADS} --threads-per-core=1 /my/exe/path/xhpcg
```

### JUBE

To adapt the provided JUBE script to your specific benchmarking needs, certain parameters in `benchmark/jube/hpcg.yml` as well as `benchmark/jube/stages_include.yml` have to be modified. The benchmark archive includes an overview of all relevant parameters of the JUBE script within the `README.md` file.

Modules for `JUBE` and `Python` need to be loaded and the current working directory is the benchmark's base directory. The following JUBE examples contain several tags, which are listed below.

```
module load Python JUBE

# s24      Stages/2024
# jrdc     JURECA-DC
# ips      ICX+ParaStationMPI
# cpu      HPCG v3.1 with CPU support (x86_64)
jube run benchmark/jube/hpcg.yml -t s24 jrdc ips cpu

# s22      Stages/2022
# jwb      JUWELS booster
# go       GCC+OpenMPI
# cuda     HPCG with Nvidia GPU support
jube run benchmark/jube/hpcg.yml -t s22 jwb go cuda

jube continue benchmark/jube/bench_run --id <JUBE ID>
jube analyse benchmark/jube/bench_run --id <JUBE ID>
jube result benchmark/jube/bench_run --id <JUBE ID>
```

#### Tags

The following JUBE tags are available:

|               Tag               |                                                       Purpose                                                        |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `cpu`                           | execute benchmark on CPU                                                                                             |
| `cuda`                          | execute benchmark on Nvidia CUDA device                                                                              |
| `rocm`                          | execute benchmark on AMD ROCm device                                                                                 |
| `nvidia`                        | Nvidia open source benchmark on CPU (Grace) and/or CUDA (Hopper), on CUDA (x86_64)                                   |
| `noref`                         | speeds up Nvidia open source benchmark jobs by skipping CPU reference calculation                                    |
| `binary`                        | Use pre-compiled binary (CPU (not nvidia-hpcg) and ROCm)                                                             |
| `s22`/`s23`/`s24`               | Use JSC software stage Stages/2022 / Stages/2023 / Stages/2024                                                       |
| `jrdc`/`jwb`/`jedi`             | Target system: JURECA-DC / JUWELS booster / JEDI                                                                     |
| `rack`/`cell`                   | Adds Slurm constraint to spawn job within specified rack or cell only                                                |
| `go`/`gps`/`gi`/`io`/`ips`/`ii` | Select toolchain (GCC+OpenMPI / GCC+ParaStationMPI / GCC+IntelMPI / ICX+OpenMPI / ICX+ParaStationMPI / ICX+IntelMPI) |


## Included Files

|      File name              |                                                Purpose                                                |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| `hpcg.yml`                  | Main YAML file for HPCG run with JUBE                                                                 |
| `stages_include.yml`        | Default compilers and modules for HPCG run                                                            |
| `submit.jedi.in`            | Slurm jobscript template for benchmark runs on JEDI                                                   |
| `hpcg.dat.in`               | Configuration file template for HPCG run                                                              |
| `hpcg-cpu.dat`              | Configuration file for HPCG CPU run                                                                   |
| `hpcg-cuda.dat`             | Configuration file for HPCG CUDA run                                                                  |
| `hpcg-rocm.dat`             | Configuration file for HPCG ROCm run                                                                  |
| `prepare_cpu.sh`            | Bash script for cloning HPCG GitHub repository                                                        |
| `prepare_cuda.sh`           | Bash script for modifying path to `xhpcg` binary in `hpcg.sh` extracted from Nvidia's container image |
| `prepare_nvidia.sh`         | Bash script for cloning Nvidia opensource HPCG GitHub repository                                      |
| `prepare_rocm.sh`           | Bash script for cloning rocHPCG source repository                                                     |

## Results

Below, sample results are quoted as achieved during creation of this benchmark.

| System |  Type  | Modules  | Nodes | Taskspernode | threadspertask | GF_Total |
|--------|--------|----------|-------|--------------|----------------|----------|
| JRDC   | x86_64 | 2022-gps |    16 |           32 |              4 |  556.524 |
| JRDC   | x86_64 | 2022-gps |    16 |           64 |              2 |  620.208 |
| JRDC   | cuda   | 2022-go  |    16 |            4 |             16 |  14363.0 |


* Type x86_64: JURECA-DC, AMD EPYC; local grid dimension x=256, y=128, z=128; time: 600 sec
* Type cuda: JURECA-DC, Nvidia A100-40; local grid dimension x=512, y=432, z=304; time: 600 sec

## Commitment

The metric of the HPCG benchmark, `GF_Total`, is to be given for execution on 1 node, 4 nodes, and all nodes of the system. The latter is used for quantitative assessment, while the performance on 1 and 4 nodes is taken into consideration for the qualitative assessment. The rules as outlined above are to be followed.  
It is the intention to submit a full-system HPCG benchmark execution to the HPCG list.
