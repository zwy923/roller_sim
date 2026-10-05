# 实验文件入口

当前工作是 **S7：固定基线的逐块独立测量验证**。结论见 [实验记录](../docs/EXPERIMENTS.md)，指标定义见 [计量段设计](../designs/station/DESIGN.md#独立测量验收口径)。

| 目录 | 用途 |
|---|---|
| [s7/](s7/README.md) | 当前基线的任务、分析和原始结果；结论数据在 `s7/results/out_valid_b/`、`s7/results/out_valid_c/` |
| [s6/](s6/README.md) | 历史结构与控制对比；`s6_run.py` 仍是 S6、S7 共用跑批入口 |

每节目录保留脚本和 README；`jobs/` 放保存的任务清单，`results/` 放原始结果，`snapshots/` 放历史源码，`archive/` 放过程记录。`out*` 和 `candidate*` 保留原版本名，便于对应历史表格。

在项目根目录运行：

```powershell
# 读取现有结果，不重跑仿真
python experiments/s7/s7_analyze.py experiments/s7/results/out_valid_b experiments/s7/results/out_valid_c

# 新试验先放 runs/，确认需要长期保存后再归入相应实验
python experiments/s7/s7_jobs.py --seeds 7101-7160 --out runs/s7_jobs.json
python experiments/s6/s6_run.py --jobs runs/s7_jobs.json --out runs/s7 --workers 10
python experiments/s7/s7_analyze.py runs/s7 --list
```

任务的 `root` 相对该 JSON 所在目录解析；未指定时使用项目当前代码。历史完整结果及源码快照原样保留，文档中的旧短目录名现在分别位于 `results/` 和 `snapshots/`。归档脚本中的旧路径只保留作历史记录。
