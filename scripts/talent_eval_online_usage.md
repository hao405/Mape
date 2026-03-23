# `talent_eval_online.py` 使用说明

## 1. 脚本作用

`Mape/scripts/talent_eval_online.py` 用于批量评测 TALENT 数据集，支持两种模式：

- 单模型评测：传 `--model_path`
- 在线监听 checkpoint 目录：传 `--models_dir`

脚本会把数据集切分到多张 GPU 上并行评测，并为每个模型写出：

- `<outdir>/<model_tag>/talent_detailed.txt`
- `<outdir>/<model_tag>/talent_summary.txt`
- `<outdir>/all_models_summary.tsv`

---

## 2. 当前 KV cache 行为

这个脚本已经适配了 TabICL 的 KV cache 版本。

默认情况下：

- `KV cache` 默认开启
- `fit()` 阶段会为训练集预建训练侧 KV cache
- `predict()` / `predict_proba()` 阶段只处理 test 数据

这和旧版本脚本最大的区别是：

- 旧逻辑更像“每次预测都把 train + test 一起重新跑”
- 新逻辑在支持场景下会复用训练侧上下文，减少重复计算

### 2.1 KV cache 适用范围

KV cache 只适用于普通分类路径，也就是：

- `n_classes <= model.max_classes`

如果数据集类别数超过模型原生支持上限：

- 默认行为：自动退回非 KV cache 路径
- 如果传 `--kv_cache_fail_on_unsupported`：直接报错，不做自动退回

### 2.2 `_dataset_cache` 和 `KV cache` 的区别

这两个缓存不是一回事。

`_dataset_cache`：

- 脚本层缓存
- 缓存的是预处理后的 `X_train / y_train / X_test / y_test`
- 默认目录是 `<outdir>/_dataset_cache`

`KV cache`：

- 分类器内部缓存
- 缓存的是训练侧推理上下文
- 不单独落盘成用户可见文件

---

## 3. 运行前准备

### 3.1 目录要求

通常要求：

- 可以访问模型 checkpoint
- 可以访问 TALENT 数据根目录
- 当前环境能导入 `tabicl`
- 当前环境能使用 GPU

### 3.2 数据目录

`--data_root` 下每个子目录视为一个数据集目录。脚本支持：

- 显式 train / val / test 文件
- 单表文件启发式拆标签

如果存在 `val` 且脚本常量 `MERGE_VAL=True`，会把验证集合并进训练集。

---

## 4. 最常用命令

### 4.1 单模型评测

```bash
python /Users/zhuhao/experiment/zijian_test/Mape/scripts/talent_eval_online.py \
  --model_path /path/to/model.ckpt \
  --data_root /path/to/datasets \
  --outdir /path/to/output \
  --clf_n_estimators 32 \
  --clf_batch_size 8 \
  --clf_n_jobs 1
```

这条命令的默认行为是：

- 开启 KV cache
- 多 GPU 并行评测所有数据集目录
- 输出每个模型的详细结果和汇总结果

### 4.2 在线监听 checkpoint 目录

```bash
python /Users/zhuhao/experiment/zijian_test/Mape/scripts/talent_eval_online.py \
  --models_dir /path/to/checkpoints \
  --data_root /path/to/datasets \
  --outdir /path/to/output \
  --poll_sec 30 \
  --stable_sec 10 \
  --step_mod 100
```

这个模式下：

- 脚本会轮询 `--models_dir`
- 发现新 checkpoint 且文件稳定后立即评测
- 已评测 checkpoint 会记到 `tested_ckpts.txt`

### 4.3 关闭 KV cache

```bash
python /Users/zhuhao/experiment/zijian_test/Mape/scripts/talent_eval_online.py \
  --model_path /path/to/model.ckpt \
  --disable_kv_cache
```

适用于：

- 想和旧逻辑对齐
- 排查 KV cache 相关问题
- 明确希望所有数据集都走非 cache 路径

### 4.4 遇到 many-class 时强制失败

```bash
python /Users/zhuhao/experiment/zijian_test/Mape/scripts/talent_eval_online.py \
  --model_path /path/to/model.ckpt \
  --kv_cache_fail_on_unsupported
```

适用于：

- 你不想接受自动退回
- 想快速发现哪些数据集不满足 KV cache 条件

---

## 5. 重要参数说明

### 5.1 模型与输入输出

- `--model_path`
  - 单个 checkpoint 路径
- `--models_dir`
  - checkpoint 目录，开启在线监听模式
- `--data_root`
  - TALENT 数据根目录
- `--outdir`
  - 输出目录

`--model_path` 和 `--models_dir` 二选一。

### 5.2 KV cache 相关

- `--disable_kv_cache`
  - 关闭 KV cache
  - 默认不传，表示开启
- `--kv_cache_fail_on_unsupported`
  - 遇到 many-class / hierarchical 不支持场景时直接报错
  - 默认不传，表示自动退回非 KV cache 路径

### 5.3 推理性能相关

- `--clf_n_estimators`
  - `TabICLClassifier` 的 ensemble 数
  - 越大通常越稳，但耗时和显存压力也会上升
- `--clf_batch_size`
  - 分类器内部 batch size
  - `-1` 表示设为 `None`，一次性处理所有 ensemble
- `--clf_n_jobs`
  - 每个进程给 PyTorch 的 CPU 线程数
- `--cpu_threads`
  - 每个 GPU worker 进程的 CPU 线程上限

### 5.4 在线监听相关

- `--poll_sec`
  - 轮询 checkpoint 目录的间隔
- `--stable_sec`
  - 文件至少稳定这么久才会被视为可评测
- `--idle_exit_sec`
  - 多久没新模型就退出
- `--step_mod`
  - 只评测 `step % step_mod == 0` 的 checkpoint

### 5.5 `torch.compile` 相关

- `--use_torch_compile`
- `--torch_compile_mode`
- `--torch_compile_backend`
- `--torch_compile_fullgraph`
- `--torch_compile_dynamic`
- `--torchinductor_cache_dir`

如果当前 `TabICLClassifier` 版本支持相关参数，脚本会透传；否则会忽略。

---

## 6. 输出文件说明

### 6.1 `talent_detailed.txt`

每个模型会生成一份逐数据集明细，列包括：

- `dataset`
- `accuracy`
- `time_s`
- `train_ratio`
- `prep_s`
- `fit_s`
- `predict_s`
- `total_e2e_s`
- `kv_cache_enabled`
- `kv_cache_used`
- `kv_cache_reason`

这三个 KV cache 字段的含义：

- `kv_cache_enabled`
  - 该次评测是否请求启用 KV cache
- `kv_cache_used`
  - 该次评测是否真的走了 KV cache 路径
- `kv_cache_reason`
  - 当前状态原因

常见 `kv_cache_reason`：

- `enabled`
  - 请求启用且成功使用
- `disabled_by_cli`
  - 你传了 `--disable_kv_cache`
- `many_class_fallback`
  - 请求启用，但数据集类别数超过模型上限，自动退回非 cache 路径
- `unsupported_classifier_version`
  - 当前环境里的 `TabICLClassifier` 不支持 `use_kv_cache`
- `fit_completed_without_cache`
  - 请求启用，但 `fit()` 后没有产出 `model_kv_cache_`

### 6.2 `talent_summary.txt`

每个模型的汇总文件除了平均 accuracy / 时间外，还会新增：

- `KV cache requested datasets`
- `KV cache used datasets`
- `KV cache fallback datasets`
- `KV cache requested ratio`
- `KV cache used ratio`
- `KV cache reason counts`

### 6.3 `all_models_summary.tsv`

这是跨模型总表，保留原有结构：

- `model_name`
- `total_datasets`
- `average_accuracy`
- `total_time_s`
- `average_time_s`
- `average_train_ratio`

这个文件不展开逐数据集的 KV cache 明细；KV cache 细节以每个模型目录下的 `talent_summary.txt` 为准。

---

## 7. 推荐使用方式

### 7.1 默认推荐

推荐先用默认配置直接跑：

- 不传 `--disable_kv_cache`
- `--clf_batch_size` 先用 `8`
- `--clf_n_jobs` 先用 `1`

这是最稳妥的起点。

### 7.2 如果显存紧张

优先调整：

- 降低 `--clf_batch_size`
- 关闭 `--use_torch_compile`

KV cache 虽然能减少 test 阶段重复计算，但 `fit()` 阶段构建 cache 本身仍然会占用显存。

### 7.3 如果数据集类别很多

默认脚本会自动退回非 KV cache 路径，所以：

- 结果不会因为 KV cache 不支持而错误
- 只是对应数据集不会享受到 KV cache 加速

如果你要严格筛出不支持的数据集，就加：

```bash
--kv_cache_fail_on_unsupported
```

---

## 8. 常见问题

### 8.1 为什么 `kv_cache_enabled=true`，但 `kv_cache_used=false`？

通常有三种情况：

- 你当前数据集触发了 `many_class_fallback`
- 当前环境里的 `TabICLClassifier` 版本不支持 `use_kv_cache`
- `fit()` 结束后没有构建出 `model_kv_cache_`

直接看 `kv_cache_reason` 即可。

### 8.2 为什么 `fit` 变慢了？

这是正常现象。

开启 KV cache 后：

- `fit()` 不只是准备数据
- 还会预建训练侧 KV cache

因此：

- `fit` 时间可能上升
- `predict` 时间通常下降

### 8.3 总体会不会更快？

通常在下面场景更容易收益明显：

- 训练集较大
- ensemble 较大
- 同一训练集上重复做多次 test 推理

但总收益仍然受这些因素影响：

- `train_size`
- `n_estimators`
- `clf_batch_size`
- 类别数
- GPU 显存

### 8.4 如何确认脚本到底用了哪个缓存？

看两类日志和输出：

- 日志中的 `_dataset_cache`
  - 这是脚本层数据缓存
- 结果文件中的 `kv_cache_*`
  - 这是分类器内部 KV cache 状态

不要把两者混在一起理解。

---

## 9. 排查建议

如果你怀疑新路径有问题，建议按这个顺序排查：

1. 先跑 `--help`，确认你实际用的是当前脚本版本
2. 先用 `--disable_kv_cache` 跑一遍，确认基线结果正常
3. 再去掉 `--disable_kv_cache`，对比 `talent_detailed.txt`
4. 重点检查 `kv_cache_used` 和 `kv_cache_reason`
5. 如果某些数据集报错，再决定是否加 `--kv_cache_fail_on_unsupported`

---

## 10. 一句话结论

如果你不想做额外决策，直接按默认方式运行就行：

- KV cache 默认开启
- many-class 默认自动退回非 cache 路径
- 详细状态会写进 `talent_detailed.txt` 和 `talent_summary.txt`
