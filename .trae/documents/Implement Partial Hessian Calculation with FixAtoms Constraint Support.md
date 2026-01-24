我将在 MACE 中实现部分 Hessian（Partial Hessian）计算功能，以支持带有固定原子约束的高效振动分析。

### 1. 修改 `mace/modules/utils.py`
- 更新 `compute_hessians_vmap` 和 `compute_hessians_loop` 函数，使其接受可选的 `indices` 参数。
    - 如果提供了 `indices`，则仅为指定的原子构建基向量 `I_N`（形状为 `[3*M, 3*N_total]`）。
    - 这将 VJP（向量-雅可比积）计算限制在必要的力分量上，从而减少计算开销。
- 更新 `get_outputs` 函数以接受 `hessian_indices` 参数，并将其传递给 Hessian 计算函数。

### 2. 修改 `mace/modules/models.py`
- 更新 `MACE.forward` 和 `ScaleShiftMACE.forward` 方法，使其接受 `hessian_indices` 参数。
- 将此参数向下传递给 `get_outputs`。

### 3. 修改 `mace/calculators/mace.py`
- 在 `MACECalculator.get_hessian` 中：
    - 检查 `atoms.constraints` 中是否存在 `FixAtoms` 约束。
    - 如果存在，提取固定原子的索引并确定“自由”（非固定）原子的索引。
    - 将 `hessian_indices=free_indices` 传递给模型调用。
    - 通过将计算出的行填充到初始化为零的矩阵中（保持固定原子的行为零），重构完整的 `(3N, 3N)` Hessian 矩阵。
    - 这样既能保证输出形状与 ASE 的预期兼容，又能利用部分计算带来的加速。

该实现允许用户传入带有 `FixAtoms` 约束的 ASE `Atoms` 对象，MACE 将自动优化 Hessian 计算，仅计算非固定原子的导数。