# Mape KV Cache 改动详解

## 1. 文档目的

本文档完整记录本次在 `Mape` 中引入 TabICLv2 风格 KV cache 的改动，覆盖：

- 改动目标与边界
- 新增数据结构
- 各模块的具体代码变化
- 训练/预测阶段的 cache 生命周期
- 张量形状与设备流转
- 测试与验证结果
- 当前限制与后续可扩展方向

这份文档对应的是当前工作区里已经实现的版本，而不是抽象方案。

---

## 2. 改动目标

### 2.1 目标

本次改动的核心目标是把 `Mape` 的分类推理从：

1. 每次 `predict_proba()` 都把 `train + test` 拼起来重新跑完整模型

改为：

1. `fit()` 阶段预计算训练侧上下文并写入 cache
2. `predict_proba()` 阶段只处理 test 样本
3. 训练侧上下文通过 cache 提供，不再重复编码

### 2.2 改动收益

在“同一训练集，多次预测不同测试集”的场景下，能显著减少重复计算，特别是：

- 列嵌入阶段的 Set Transformer
- ICL 阶段的训练侧 attention K/V 投影

### 2.3 本次明确不做的内容

本次实现严格限定在 `kv cache`，不包含：

- `repr cache`
- cache 序列化保存/加载
- v2 更大的 inference/offload 重构
- class shuffle 语义切换
- many-class / hierarchical classification 的缓存支持

---

## 3. 改动范围总览

本次修改覆盖以下文件：

- `src/tabicl/model/kv_cache.py`：新增 cache 数据结构
- `src/tabicl/model/attention.py`：支持 cached K/V 和返回新 K/V
- `src/tabicl/model/layers.py`：attention block / ISAB block 支持缓存路径
- `src/tabicl/model/encoders.py`：`Encoder` / `SetTransformer` 支持 `forward_with_cache`
- `src/tabicl/model/inference.py`：`InferenceManager` 支持 `KVCache` 切片与回写
- `src/tabicl/model/embedding.py`：列嵌入支持 store/use cache
- `src/tabicl/model/learning.py`：ICL 层支持 store/use cache
- `src/tabicl/model/tabicl.py`：模型级 `forward_with_cache`
- `src/tabicl/sklearn/preprocessing.py`：支持 `mode="train"|"test"|"both"`
- `src/tabicl/sklearn/classifier.py`：`fit()` 构建 cache，`predict_proba()` 走 cache 路径
- `tests/test_sklearn.py`：新增 cache 回归测试

---

## 4. 整体设计思路

### 4.1 高层原则

整个实现遵循两个原则：

1. 不改原有无 cache 路径的预测语义
2. cache 只缓存训练侧可复用上下文，不缓存 test 输出

### 4.2 cache 的两条主线

本次 cache 实际分成两层：

#### A. Column Embedding cache

缓存的是 Set Transformer 中每个 `InducedSelfAttentionBlock` 第二段 attention 的 K/V。

原因：

- 第一段 attention 是 `inducing points -> training features`
- 第二段 attention 是 `features -> hidden`
- 一旦训练侧 `hidden` 固定，第二段 attention 的 K/V 可以直接复用给 test features

#### B. ICL cache

缓存的是 ICL `Encoder` 每一层来自训练侧样本的 K/V 投影。

原因：

- 训练标签已经写入训练表示
- 后续 test token 只需要作为 query 去读这些 K/V
- 不必重复跑训练侧上下文编码

### 4.3 为什么 `RowInteraction` 不缓存

`RowInteraction` 仍然保持原样，不做 cache，原因是：

- 本次目标是最小化改动并优先拿到主要收益
- 列嵌入和 ICL 是训练集重复计算里最值得缓存的两段
- `RowInteraction` 当前没有像 Set Transformer / ICL Encoder 那样天然的“训练侧 K/V 可直接复用”的结构

---

## 5. 新增数据结构：`src/tabicl/model/kv_cache.py`

这是本次改动的基础文件。

### 5.1 `KVCacheEntry`

表示一个 attention 位点的一组缓存：

- `key: Optional[Tensor]`
- `value: Optional[Tensor]`

#### 作用

- 存一个 block / 一层 attention 的 K/V
- 支持按 batch 维切片
- 支持移动设备
- 支持按 batch 维拼接

#### 关键方法

- `is_valid()`
  - 判断当前 entry 是否同时拥有 `key` 和 `value`
- `__getitem__(indices)`
  - 对 K/V 做 batch slice
- `__setitem__(indices, other)`
  - 把一个 batch 子块写回当前 entry
- `to(device)`
  - 搬到目标设备
- `concat(entries, dim=0)`
  - 沿 batch 维拼接多个 entry

### 5.2 `KVCache`

表示“若干层/若干 block 的 KV 缓存集合”：

- `kv: Dict[int, KVCacheEntry]`

这里的 key 通常是：

- ICL encoder 的 layer index
- Set Transformer 中 ISAB block 的 block index

#### 关键方法

- `is_populated()`
  - 判断 cache 是否已经写入有效 entry
- `__getitem__(indices)`
  - 对整个 cache 的所有 entry 做 batch slice
- `__setitem__(indices, other)`
  - 把一个 batch 子 cache 写回总 cache
- `to(device)`
  - 把整份 cache 移到目标设备
- `concat(caches, dim=0)`
  - 把多个 batch 的 cache 合并
- `preallocate(reference, batch_shape, device)`
  - 在总 cache 上按参考 cache 的形状预分配张量

`preallocate` 是 `InferenceManager` 批处理回写时最关键的一步。

### 5.3 `TabICLCache`

模型级总 cache 容器：

- `col_cache`
- `icl_cache`
- `train_shape`
- `num_classes`

#### 为什么要有顶层容器

因为 `predict_proba()` 只传 test，但模型仍然需要知道：

- 训练集构建出的列嵌入 cache
- ICL 层 cache
- 当前训练集对应的类别数

#### 关键方法

- `is_empty()`
- `slice_batch(start, end)`
- `__getitem__(indices)`
- `to(device)`
- `concat(caches, dim=0)`

`slice_batch()` 主要被 `classifier._batch_forward_with_cache()` 用来按 estimator batch 切 cache。

---

## 6. Attention 层改动：`src/tabicl/model/attention.py`

### 6.1 改动前

`multi_head_attention_forward()` 只支持标准路径：

- 输入 `query`, `key`, `value`
- 一次性投影出 `q`, `k`, `v`
- 做 attention

### 6.2 改动后新增的能力

#### 新增参数

- `key: Optional[Tensor] = None`
- `value: Optional[Tensor] = None`
- `cached_kv: Optional[KVCacheEntry] = None`
- `need_kv: bool = False`

#### 新增两条执行路径

##### 路径 1：标准路径

当 `cached_kv is None`：

- 仍然从 `query/key/value` 正常计算
- 若 `need_kv=True`，会把新算出的 `k` 和 `v` 一起返回

这条路径用于 `store_cache=True` 的阶段。

##### 路径 2：缓存路径

当 `cached_kv is not None`：

- 只对 `query` 做 Q 投影
- 直接使用缓存中的 K/V
- 不再重复计算训练侧 K/V

这条路径用于 `use_cache=True` 的阶段。

### 6.3 RoPE 在这里的处理

#### 标准路径

- `q` 旋转
- `k` 旋转

#### 缓存路径

- 只旋转 `q`
- 不再旋转 cached `k`

原因是 cached `k` 在写入 cache 之前已经旋转过了，重复旋转会导致位置编码错误。

### 6.4 为什么保留 `attn_mask=int` 路径

旧代码里 ICL 用过 `attn_mask=train_size` 的特殊分段逻辑。虽然 cache 路径已经不再依赖它，但为了不破坏已有非 cache 路径，这部分逻辑继续保留。

---

## 7. Layer 层改动：`src/tabicl/model/layers.py`

这一层是本次改动里最关键的中间层，因为它把底层 attention 能力暴露成 block 级接口。

### 7.1 `MultiheadAttention`

新增支持：

- `cached_kv`
- `need_kv`

作用：

- 把 `attention.py` 的新能力直接暴露给上层 block

### 7.2 `MultiheadAttentionBlock`

#### 新增参数

- `cached_kv`
- `train_size`
- `need_kv`

#### 核心变化

原先 block 只能处理：

- `q, k, v`
- 或默认 `k = q, v = q`

现在 block 还能处理：

##### A. `train_size` 路径

表示：

- `q` 是完整序列
- `k/v` 只取前 `train_size` 个位置

这替代了 ICL 过去通过 `attn_mask=int` 间接表达“只有训练样本做上下文”的做法。

##### B. `cached_kv` 路径

表示：

- `q` 是当前输入
- `k/v` 从 cache 读

##### C. `need_kv=True`

表示：

- block 在前向时除了返回输出，还返回本层新生成的 `k_proj`, `v_proj`
- 上层可以把它写入 cache

#### 为什么这里要支持 `train_size`

因为 ICL cache 不是把完整 `train+test` 都当上下文，而是：

- store 阶段：完整输入前向，但 K/V 只来源于训练区间
- use 阶段：test 只作为 query，K/V 来自 cache

### 7.3 `InducedSelfAttentionBlock`

这是列嵌入缓存的核心位置。

#### 新增方法 1：`induced_attention_with_cache(...)`

两种模式：

##### `store_cache=True`

1. 用训练区间样本生成 `hidden`
2. 让 `src -> hidden` 的第二段 attention 返回 `out, k_proj, v_proj`
3. 把 `k_proj/v_proj` 写到 `col_cache.kv[block_idx]`

##### `use_cache=True`

1. 跳过重新生成训练侧 `hidden`
2. 直接用 `col_cache.kv[block_idx]`
3. 让当前 `src` 作为 query 去读缓存

#### 新增方法 2：`forward_with_cache(...)`

包装了：

- `use_cache == store_cache` 的合法性检查
- `train_size` 必填检查
- `skip_value` 逻辑兼容

#### 为什么缓存的是第二段 attention 的 K/V

因为：

- 第一段 attention 的输出 `hidden` 依赖训练数据
- 第二段 attention 使用 `hidden` 作为 K/V
- 对 test 来说，可复用的是第二段 attention 所需的 K/V

---

## 8. Encoder 层改动：`src/tabicl/model/encoders.py`

### 8.1 `Encoder`

原先只支持标准 `forward()`。

现在新增：

- `forward(..., train_size=None)`
- `forward_with_cache(...)`

#### `forward(..., train_size=...)`

让非 cache 的 ICL 路径也可以显式表达：

- 完整序列作为 query
- 仅训练部分作为 K/V

#### `forward_with_cache(...)`

对每一层 block：

- `store_cache=True`：
  - 调 block 的 `need_kv=True`
  - 写 `icl_cache.kv[layer_idx]`
- `use_cache=True`：
  - 把 `icl_cache.kv[layer_idx]` 作为 `cached_kv`

### 8.2 `SetTransformer`

新增 `forward_with_cache(...)`。

逻辑很直接：

- 按 block 遍历
- 每个 block 调 `InducedSelfAttentionBlock.forward_with_cache(...)`

---

## 9. InferenceManager 改动：`src/tabicl/model/inference.py`

这是本次改动中最容易被忽略、但实际上非常关键的一层。

### 9.1 改动前的问题

原来的 `InferenceManager` 只知道怎么处理：

- `Tensor`
- 普通标量或对象

它不知道 `KVCache` 是什么，所以一旦前向函数参数里包含 cache：

- 要么整块原样透传
- 要么无法按 batch 正确切分

这会直接导致：

- store cache 时无法按 mini-batch 累积回总 cache
- use cache 时无法按 estimator batch slice cache

### 9.2 新增能力

#### 新增 `prepare_input_value()`

统一处理单批输入：

- `Tensor` -> 移到执行设备
- populated `KVCache` -> 移到执行设备
- empty `KVCache` -> 保持对象本身

#### 修改 `create_multidim_batches()`

对于 `KVCache`：

##### 若 cache 已填充

- 对 cache 按 `slice_tuple` 切 batch
- 再移到执行设备

##### 若 cache 为空

- 给当前 mini-batch 一个新的空 `KVCache()`
- 由当前前向过程把 K/V 写进去

### 9.3 新增 mini-batch cache 回写逻辑

在 `__call__()` 主循环里新增：

1. 识别 `store_cache_keys`
2. 每个 mini-batch 跑完后检查对应 cache 是否被填充
3. 若总 cache 还没分配张量，则用当前 batch cache 做 `preallocate`
4. 再把当前 batch cache 回写到总 cache 对应切片

### 9.4 OOM 重试时的 cache 处理

如果当前 batch OOM：

- 原先只会减小 `batch_size`
- 现在还会清空已经部分写入的 cache

这样可以避免“前半截 cache 是旧结果，后半截 cache 还没写”的残缺状态被复用。

---

## 10. Column Embedding 改动：`src/tabicl/model/embedding.py`

### 10.1 新增 `_compute_embeddings_with_cache(...)`

做的事情：

1. `in_linear`
2. `tf_col.forward_with_cache(...)`
3. `out_w/out_b`
4. `features * weights + biases`

也就是说，缓存并没有改变列嵌入输出公式，只是让中间的 Set Transformer 可以 store/use cache。

### 10.2 新增 `forward_with_cache(...)`

职责：

- 校验 `use_cache` / `store_cache`
- 校验 `train_size`
- 配置 `InferenceManager`
- 构造 `features`
- 通过 `InferenceManager` 执行 `_compute_embeddings_with_cache(...)`

### 10.3 这里为什么不再处理 `feature_shuffles`

cache 路径下，`predict_proba()` 已经改成：

- `EnsembleGenerator.transform(mode="test")`
- 每个 estimator 的 test 输入都已经是独立的 feature 顺序

所以 cache 路径不再需要像旧路径一样在 `ColEmbedding` 内部做“先算第一张表，再映射 shuffle”这种优化。

---

## 11. ICL 层改动：`src/tabicl/model/learning.py`

### 11.1 非 cache 路径也改成了 `train_size`

原来 `_icl_predictions()` 用的是：

- `self.tf_icl(R, attn_mask=train_size)`

现在改成：

- `self.tf_icl(R, train_size=train_size)`

原因：

- cache 路径的语义是显式 “query=全序列 / K,V=训练区间”
- 非 cache 路径也切到同一套表达方式，避免两套语义并行

### 11.2 新增 `_icl_predictions_with_cache(...)`

#### store 阶段

1. 把 `y_train` 编码后加到 `R[:, :train_size]`
2. 调 `self.tf_icl.forward_with_cache(..., store_cache=True)`
3. 由 encoder 层缓存每层训练侧 K/V

#### use 阶段

1. 不再重复注入 `y_train`
2. 只把当前 test 表示作为 query
3. 训练侧上下文从 `icl_cache` 读取

### 11.3 新增 `forward_with_cache(...)`

职责：

- 参数合法性检查
- many-class 拒绝缓存
- 配置 `InferenceManager`
- 调用 `_icl_predictions_with_cache(...)`
- 在 store 阶段裁掉训练部分输出
- 在分类任务里裁到 `num_classes`

### 11.4 many-class 为什么这里直接报错

当前 `Mape` 的 many-class 路径依赖 hierarchical classification。

这个过程不是单次固定上下文前向，而是：

- 不同节点递归构建不同训练子集
- 多轮不同的 ICL 前向

因此它和本次“固定训练上下文 -> 复用 K/V”的 cache 设计不兼容，所以直接拒绝。

---

## 12. 模型总入口改动：`src/tabicl/model/tabicl.py`

### 12.1 新增 `self._cache`

`TabICL` 现在内部持有：

- `self._cache: Optional[TabICLCache]`

### 12.2 新增 `clear_cache()`

作用很简单：

- 清掉当前模型实例上挂着的 cache 引用

这个方法主要被 sklearn 层在 batch 构建 cache 后调用，避免上一批 estimator 的 cache 残留在模型实例上。

### 12.3 新增 `forward_with_cache(...)`

这是模型级 cache 主入口。

#### 支持两种模式

##### A. `store_cache=True`

要求：

- 必须给 `X_train`, `y_train`

流程：

1. 初始化 `TabICLCache`
2. 若 `X_test` 存在，则拼成 `X_train + X_test`
3. 列嵌入走 `forward_with_cache(..., store_cache=True)`
4. 行交互照常走
5. ICL 走 `forward_with_cache(..., store_cache=True)`
6. 若没有 `X_test`，说明只是建 cache，最终返回 `None`

##### B. `use_cache=True`

要求：

- 必须给 `X_test`
- 必须已有 `self._cache` 或外部传入 `cache`

流程：

1. `X = X_test`
2. 列嵌入从 `col_cache` 读
3. 行交互照常走
4. ICL 从 `icl_cache` 读

#### 外部 `cache` 参数的作用

在 sklearn 侧，`predict_proba()` 会把按 batch slice 后的 `cache_subset` 传进来。此时：

- `use_cache=True`
- `store_cache=False`
- 模型直接使用这份外部 cache

---

## 13. EnsembleGenerator 改动：`src/tabicl/sklearn/preprocessing.py`

### 13.1 改动前

`transform(X)` 只能返回：

- `train + test` 拼接后的完整 prompts

### 13.2 改动后新增 `mode`

现在支持：

- `mode="train"`
- `mode="test"`
- `mode="both"`

### 13.3 三种模式的语义

#### `mode="train"`

返回：

- 每个 normalization method 下，已经预处理好的训练集
- 每个 estimator 对应的 feature shuffle
- 每个 estimator 对应的 class shift 后标签

用途：

- `classifier.fit()` 构建训练侧 cache

#### `mode="test"`

返回：

- 仅 test 数据
- 已经应用 feature shuffle
- 不返回标签

用途：

- `predict_proba()` 在 use cache 路径里只前向 test

#### `mode="both"`

保留旧行为：

- 返回 `train + test` 的拼接输入
- 用于无 cache 路径

### 13.4 为什么保留当前 `class_shift` 设计

因为 `Mape` 原先的 ensemble 聚合逻辑就是：

- 对训练标签做 cyclic shift
- 预测后再 reverse shift

本次只引入 KV cache，不同时引入 v2 的 class shuffle 语义变更，避免行为漂移。

---

## 14. sklearn 分类器改动：`src/tabicl/sklearn/classifier.py`

### 14.1 新增构造参数 `use_kv_cache`

```python
TabICLClassifier(..., use_kv_cache=False)
```

这样可以直接兼容已有脚本：

- `scripts/talent_eval_online.py`

因为那个脚本是通过构造参数签名探测 `use_kv_cache` 的，而不是通过 `fit(..., kv_cache=...)`。

### 14.2 `fit()` 的变化

在原流程：

1. 数据校验
2. 加载模型
3. 标签编码
4. 特征预处理
5. ensemble 配置

之后，现在新增：

6. 若 `self.use_kv_cache=True`，执行 `_build_kv_cache()`

### 14.3 `_build_kv_cache()`

流程：

1. `self.ensemble_generator_.transform(X=None, mode="train")`
2. 拿到每个 normalization method 下所有训练 prompts
3. 按 `batch_size` 分批
4. 每个 batch 调用：

   ```python
   self.model_.forward_with_cache(
       X_train=...,
       y_train=...,
       use_cache=False,
       store_cache=True,
   )
   ```

5. 取出 `self.model_._cache`
6. 调 `self.model_.clear_cache()`
7. 最后把所有 batch 的 cache 用 `TabICLCache.concat(...)` 合并

最终得到：

- `self.model_kv_cache_[norm_method]`

### 14.4 `_batch_forward_with_cache()`

这是 cache 预测阶段的新方法。

流程：

1. test 数据按 `batch_size` 切分
2. 从总 cache 中 `slice_batch(offset, offset + bs)`
3. 调：

   ```python
   self.model_.forward_with_cache(
       X_test=X_batch,
       cache=cache_subset,
       ...
   )
   ```

4. 收集输出

### 14.5 `predict_proba()` 的分支逻辑

#### 无 cache

继续旧路径：

- `transform(mode="both")`
- 前向 `train + test`
- reverse `class_shift`
- ensemble 聚合

#### 有 cache

现在改成：

- `transform(mode="test")`
- 只前向 test
- 从 `self.model_kv_cache_` 取 cache
- reverse `class_shift`
- ensemble 聚合

### 14.6 many-class 的处理

`fit()` 中新增显式判断：

若：

- `use_kv_cache=True`
- `n_classes > model_.max_classes`

则直接抛错。

这保证：

- 不会误进 hierarchical cache 不支持的场景

---

## 15. 测试改动：`tests/test_sklearn.py`

### 15.1 新增 mock `_load_model()`

因为真实 `TabICLClassifier._load_model()` 依赖 Hugging Face checkpoint，所以测试里新增：

- `_mock_load_model()`

它会构造一个小尺寸 `TabICL` 模型并固定随机种子，避免下载真实 checkpoint。

### 15.2 新增测试 1：cache 与无 cache 结果一致

`test_predict_proba_matches_with_kv_cache`

验证：

- `use_kv_cache=False` 的 `predict_proba`
- `use_kv_cache=True` 的 `predict_proba`

在同一数据上输出一致：

- `rtol=1e-4`
- `atol=1e-4`

### 15.3 新增测试 2：many-class cache 禁用

`test_many_class_kv_cache_is_rejected`

验证：

- 当 `max_classes=2`
- 实际训练数据有 3 类
- 且 `use_kv_cache=True`

会抛出预期 `ValueError`

---

## 16. 训练/预测阶段完整数据流

### 16.1 `fit(..., use_kv_cache=False)`

和旧版本一致：

1. 训练集预处理
2. ensemble 配置
3. 不建 cache

### 16.2 `fit(..., use_kv_cache=True)`

新增路径：

1. 训练集预处理
2. ensemble 配置
3. 每个 normalization method 下构造训练 prompts
4. 按 estimator batch 前向 `forward_with_cache(..., store_cache=True)`
5. 把每个 batch 的 `TabICLCache` 合并成总 cache

### 16.3 `predict_proba(..., use_kv_cache=False)`

旧路径：

1. test 预处理
2. `transform(mode="both")`
3. 模型输入是 `train + test`
4. 输出聚合

### 16.4 `predict_proba(..., use_kv_cache=True)`

新路径：

1. test 预处理
2. `transform(mode="test")`
3. 模型输入只有 `X_test`
4. 训练侧上下文来自 `model_kv_cache_`
5. 输出聚合

---

## 17. 张量形状说明

### 17.1 Column cache 的 K/V 形状

当列嵌入前向时，`features` 的 batch 维是：

- `(B, H+C)`

所以列 cache 的 K/V 形状实际是：

```python
(*batch_shape, num_heads, seq_len, head_dim)
```

也就是：

```python
(B, H+C, num_heads, seq_len, head_dim)
```

这也是为什么 `InferenceManager` 必须支持对 `KVCache` 按多维 batch slice。

### 17.2 ICL cache 的 K/V 形状

ICL 表示的 batch 维只有：

- `B`

所以 ICL cache 的形状是：

```python
(B, num_heads, train_size, head_dim)
```

按 layer index 存在 `icl_cache.kv[layer_idx]` 中。

---

## 18. 设备与内存行为

### 18.1 单批路径

若 `InferenceManager` 判断无需自动分批：

- populated cache 直接搬到执行设备
- empty cache 原地传入，前向中写入

### 18.2 多批路径

#### use cache

- 每个 mini-batch 取当前 batch 对应的 cache slice

#### store cache

- 每个 mini-batch 分到一个空 cache
- 前向后回写到总 cache

### 18.3 OOM 恢复

若某一轮 mini-batch OOM：

- `batch_size` 减半
- 已部分写入的 cache 清空
- 从头重试

---

## 19. 本次实现的限制

### 19.1 不支持 `repr cache`

当前只缓存：

- `col_cache`
- `icl_cache`

没有缓存 row representation。

### 19.2 不支持 many-class cache

即：

- `n_classes > max_classes`

时 cache 直接报错。

### 19.3 不支持 cache 序列化

当前 cache 生命周期只在当前 Python 进程内。

### 19.4 未修改外部公开包导出

`TabICLCache` 等类型当前只在内部使用，没有从 `tabicl.__init__` 导出。

---

## 20. 验证结果

### 20.1 通过的验证

#### 代码编译

```bash
python -m compileall src/tabicl tests
```

通过。

#### 定向测试

```bash
pytest tests/test_sklearn.py -k "predict_proba_matches_with_kv_cache or many_class" -q
```

结果：

- `2 passed`

### 20.2 未完全通过的验证

```bash
pytest tests/test_sklearn.py -q
```

这条命令没有全部通过，但失败原因不是本次 KV cache 逻辑，而是测试环境的现有依赖问题：

- `_load_model()` 会触发 Hugging Face 下载
- 当前环境配置了 `socks5://127.0.0.1:7890` 代理
- `httpx` 走 SOCKS 代理时缺 `socksio`

所以完整 sklearn estimator checks 在进入模型逻辑前就因环境问题失败。

---

## 21. 对后续维护者的建议

### 21.1 如果后续要加 `repr cache`

建议优先扩展：

- `TabICLCache`
- `TabICL.forward_with_cache`
- `ICLearning.forward_with_cache`

因为这三处已经是 cache 的总入口。

### 21.2 如果后续要支持 cache 序列化

建议补：

- `TabICLCache.to("cpu")`
- save/load helper
- sklearn estimator 的 `__getstate__` / `__setstate__` 风格逻辑

### 21.3 如果后续要支持 many-class cache

需要重新设计 hierarchical classification 与 cache 的关系，这不是简单放开当前判断就能工作的事。

---

## 22. 一句话总结

本次改动把 `Mape` 的分类推理从“每次预测重复编码训练上下文”升级为“训练阶段预建 KV cache、预测阶段只跑 test”，并且把这条能力从 attention、block、encoder、inference manager、model 一直到 sklearn `fit/predict` 全链路打通了。
