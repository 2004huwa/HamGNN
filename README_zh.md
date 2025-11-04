<p align="center">
  <img height="130" src="logo/logo.png"/>
</p>

## 🚀 HamGNN v2.0 使用说明（中文）

### 目录
- **项目简介**
- **环境与依赖**
  - Python 与三方库
  - OpenMX / openmx_postprocess / read_openmx
- **安装**
- **使用流程总览**
  - 训练用哈密顿量数据准备
  - 评估/预测用数据准备
  - 图数据打包（graph_data.npz）
  - 网络训练与预测（HamGNN v2.0 / v1.0）
  - 能带微调训练（二阶段）
  - 能带结构计算（串行与并行）
- **对 ABACUS 的支持**
- **对 HONPAS/SIESTA 的支持**
  - honpas_1.2_H0
  - hsxdump
  - 训练数据流程
  - 预测流程
- **配置文件说明（config.yaml）**
- **最小不可约表示建议（irreps）**
- **引用与致谢**

---

### 项目简介
HamGNN 是一个 E(3) 等变图神经网络，用于对分子与固体体系的一阶近似（TB）哈密顿量进行训练与预测，适配基于数值原子轨道的常见第一性原理软件（如 OpenMX、SIESTA、ABACUS）。项目亦支持包含自旋轨道耦合（SOC）的 SU(2) 等变哈密顿量预测，可在保证精度的同时显著加速大规模体系的电子结构计算。

---

### 环境与依赖
- **Python 版本建议**: 3.9
- **核心依赖**（简要）：`numpy`、`torch`、`torch_geometric`、`pytorch_lightning`、`e3nn`、`pymatgen`、`tensorboard`、`tqdm`、`scipy`、`yaml`。详细版本可参考仓库根目录的 `environment.yaml` 与 `setup.py`。

- **快速环境构建**：
  - 使用 Conda 环境文件：
    ```bash
    conda env create -f environment.yaml
    ```
    注意：当前 `environment.yaml` 在 SOC 训练时可能引发如下报错：
    ```
    RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    ```
  - 使用预构建环境（推荐、更稳健）：从 Zenodo 获取预构建 Conda 环境（见英文 README 链接），解压 `ML.tar.gz` 至你的 `conda/envs` 目录。

- **OpenMX 与相关工具**：
  - OpenMX 下载与基础使用：`https://www.openmx-square.org/`
  - `openmx_postprocess`：修改后的 OpenMX，用于解析并输出重叠矩阵等量，生成二进制 `overlap.scfout`
    1) 安装 GSL；2) 在 `openmx_postprocess` 目录修改 `makefile` 中 `GSL_lib`、`GSL_include`、`MKLROOT`、`CMPLR_ROOT`；3) 执行 `make` 生成 `openmx_postprocess` 与 `read_openmx`。
  - `read_openmx`：从 `overlap.scfout` 导出矩阵到 `HS.json` 的可执行程序。

---

### 安装
```bash
git clone https://github.com/QuantumLab-ZY/HamGNN.git
cd HamGNN
python setup.py install
```
升级时请先卸载旧版本并清理 `site-packages` 中残留的 `HamGNN-x.x.x-py3.9.egg/HamGNN` 目录后再安装。

安装后将得到以下命令行入口（见 `setup.py`）：
- `HamGNN1.0`：运行 v1.0 主程序
- `HamGNN2.0`：运行 v2.0 主程序
- `graph_data_gen`：OpenMX 工作流的图数据打包
- `poscar2openmx`：POSCAR→OpenMX 输入转换
- `band_cal`：串行能带计算（OpenMX/H0）

---

### 使用流程总览

#### 1) 训练用哈密顿量数据准备
1. 生成结构文件（POSCAR 或 CIF），可由分子动力学或随机微扰得到。
2. 转换为 OpenMX 输入：编辑 `utils_openmx/poscar2openmx.yaml`，执行：
   ```bash
   poscar2openmx --config utils_openmx/poscar2openmx.yaml
   ```
   将结构批量转换为 `.dat` 等 OpenMX 格式。
3. 运行 OpenMX 静态计算，获得包含哈密顿量与重叠矩阵信息的 `.scfout`。
4. 运行 `openmx_postprocess` 生成 `overlap.scfout`，其中包含与电荷密度无关的 `H0`。

#### 2) 评估/预测用数据准备
- 若已有训练好的模型，可跳过 OpenMX 自洽计算，直接将 `openmx_postprocess` 输出的 `overlap.scfout` 作为评估输入。

#### 3) 图数据打包（graph_data.npz）
1. 编辑 `utils_openmx/graph_data_gen.yaml`，关键字段：
   - `nao_max`: 14/19/26 等；与元素基函数最大数相关
   - `graph_data_save_path`: 输出目录
   - `read_openmx_path`: `read_openmx` 可执行文件路径
   - `scfout_paths`: `.scfout` 或 `overlap.scfout` 文件所在路径（可通配）
   - `soc_switch`: 是否为 SOC 数据集
2. 执行：
   ```bash
   graph_data_gen --config utils_openmx/graph_data_gen.yaml
   ```
   生成单一的 `graph_data.npz`，供网络使用。

#### 4) 网络训练与预测（v2.0 / v1.0）
- 在对应版本目录内提供示例 `config.yaml`：
  - v2.0: `HamGNN_v_2_0/config.yaml`
  - v1.0: `HamGNN_v_1_0/config.yaml`

- 训练：
  ```bash
  HamGNN2.0 --config HamGNN_v_2_0/config.yaml
  ```
  可在 `config.yaml` 的 `setup.stage` 设为 `fit` 进行训练；使用 TensorBoard 监控：
  ```bash
  tensorboard --logdir <train_dir>
  ```

- 预测：
  1) 将 `setup.stage` 设为 `test`
  2) 指定 `setup.checkpoint_path` 为训练权重
  3) 执行：
  ```bash
  HamGNN2.0 --config HamGNN_v_2_0/config.yaml
  ```

#### 5) 能带微调训练（二阶段，可选）
- 在完成哈密顿量训练后，可在 `config.yaml` 中：
  - `load_from_checkpoint: True`，并设置 `checkpoint_path`
  - 减小 `optim_params.lr`（如 `1e-4`）
  - 在 `losses_metrics` 中加入 `band_energy` 相关项并设置较小权重（如 0.01）
  - 在 `output_nets.HamGNN_out` 中启用 `calculate_band_energy`、配置 `num_k`、`band_num_control`、`k_path` 等

#### 6) 能带结构计算
- 串行：
  1) 编辑 `utils_openmx/band_cal.yaml`，关键字段：`graph_data_path`、`hamiltonian_path`（若留空则使用 `graph_data.npz` 内部哈密顿量）、`nk`、`k_path`/`auto_mode`、`nao_max` 等
  2) 执行：
     ```bash
     band_cal --config utils_openmx/band_cal.yaml
     ```

- 并行（适用于大体系）：
  1) 安装 wheel（仓库提供示例）：
     ```bash
     pip install band_cal_parallel/mpitool-0.0.1-cp39-cp39-manylinux1_x86_64.whl
     pip install band_cal_parallel/band_cal_parallel-0.1.15-py3-none-any.whl
     ```
  2) 编辑 `band_cal_parallel/band_cal_parallel.yaml`，如：
     - `graph_data_path`: `graph_data.npz` 路径
     - `hamiltonian_path`: 预测哈密顿量 `prediction_hamiltonian.npy`（可空）
     - `k_path`/`label`（可空，自动生成路径时置空）
     - `nk`: 总 k 点数
     - `nao_max`、`Ham_type`（`openmx` 或 `abacus`）
  3) 运行：
     ```bash
     mpirun -np <ncpus> band_cal_parallel --config band_cal_parallel/band_cal_parallel.yaml
     ```
  4) 建议在作业脚本中设置：`export OMP_NUM_THREADS=<ncpus_per_node>`

---

### 对 ABACUS 的支持（`utils_abacus/`）
- 提供 `abacus_postprocess`（H0 导出）、`poscar2abacus.py`（结构转换）、`graph_data_gen_abacus.py`（打包 `graph_data.npz`）。
- 为更好地拟合 HSE 哈密顿量，修复了旧脚本使用 PBE `H0` 的 `edge_index` 截断 HSE 哈密顿量的问题；旧打包脚本已弃用。

---

### 对 HONPAS/SIESTA 的支持（`utils_siesta/`）
- `honpas_1.2_H0`：修改版 HONPAS，计算非自洽 `H0` 与重叠矩阵，输出 `overlap.HSX`。
- `hsxdump`：生成 HONPAS 输出到 HamGNN 可读中间格式的二进制工具。在 `utils_siesta/hsx4.1.5` 下编译：
  ```bash
  cd utils_siesta/hsx4.1.5
  make
  ```

#### 训练数据流程
1) 结构生成（POSCAR/CIF）
2) 使用 `poscar2siesta.py` 转换为 `.fdf`
3) 运行 HONPAS 获得 `.HSX`
4) 使用 `honpas_1.2_H0` 生成 `overlap.HSX`：
   ```bash
   mpirun -np <Ncores> honpas_1.2_H0 < input.fdf
   ```
5) 运行 `graph_data_gen_siesta.py` 生成 `graph_data.npz`

#### 预测流程
1) 使用 `poscar2siesta.py` 生成 `.fdf`
2) 运行 HONPAS 生成 `overlap.HSX`
3) 使用 `predict_data_gen_siesta.py` 打包预测所需 `graph_data.npz`

---

### 配置文件说明（以 v2.0 为例：`HamGNN_v_2_0/config.yaml`）
- `setup`：
  - `stage`: `fit` 训练 / `test` 推理
  - `GNN_Net`: `HamGNNpre`（等变卷积）
  - `num_gpus`: GPU 数或索引；`precision`: 32/64；`checkpoint_path` 与 `load_from_checkpoint`/`resume`
- `dataset_params`：`graph_data_path`（文件或目录）、`batch_size`、`train/val/test_ratio`、`split_file`
- `losses_metrics`：支持 `mae/mse/rmse` 等；可按需为 `hamiltonian/band_energy/overlap` 等目标配置损失与指标
- `optim_params`：`lr`、`lr_decay`、`lr_patience`、`max/min_epochs`、`stop_patience`、`gradient_clip_val`
- `profiler_params`：`train_dir`（TensorBoard 日志与结果目录）
- `representation_nets.HamGNN_pre`：`cutoff`、`irreps_*`、`num_layers`、`num_radial`、`num_types`、`rbf_func` 等，常规无需改动
- `output_nets.HamGNN_out`：
  - `ham_type`: `openmx`/`abacus`
  - `nao_max`: 14/19/26/27/40（依据元素与基函数设置）
  - `add_H0`、`symmetrize`、`soc_switch`、`nonlinearity_type`、`calculate_band_energy`、`num_k`、`band_num_control`、`k_path`
  - 自旋相关：`spin_constrained`、`collinear_spin`、`minMagneticMoment`

---

### 最小不可约表示建议（irreps）
如需参考最小 irreps 设置，可见英文 README 中示例与输出：`17x0e+20x1o+...`。

---

### 引用与致谢
- 若本项目助力你的研究，请引用相关论文（详见英文 README 中 References）。
- 代码贡献者与项目负责人列表亦可见英文 README。

