# 实验文件入口

当前工作是 **S7：固定基线的逐块独立测量验证**。结论见 [实验记录](../docs/EXPERIMENTS.md)，指标定义见 [计量段设计](../designs/station/DESIGN.md#独立测量验收口径)。

| 目录 | 用途 |
|---|---|
| [s7/](s7/README.md) | 当前基线的任务、分析和原始结果；结论数据在 `s7/results/out_valid_b/`、`s7/results/out_valid_c/` |

`s7/` 里是脚本、README、`REPORT.md`（完整记录）和 `results/`（原始结果，保留 `out*` 原组名）。`results/` 不进 git，只有本机这一份，需要长期保存请另行备份；每批结果的摘要里记着它自己的命令行参数和源码哈希。第 S6 节（历史结构与控制对比）只留下记录 [docs/history/EXPERIMENTS_S6.md](../docs/history/EXPERIMENTS_S6.md)；当时的脚本、源码快照和任务清单在 git 标签 `before-refactor` 上（见项目 [README](../README.md#2026-10-05-结构重构)）。

在项目根目录运行：

```powershell
# 读取现有结果，不重跑仿真
python experiments/s7/s7_analyze.py experiments/s7/results/out_valid_b experiments/s7/results/out_valid_c

# 新试验先放 runs/，确认需要长期保存后再归入相应实验的 results/
python experiments/s7/s7_jobs.py --seeds 7101-7160 --out runs/s7_jobs.json
python experiments/s7/s7_run.py --jobs runs/s7_jobs.json --out runs/s7 --workers 10
python experiments/s7/s7_analyze.py runs/s7 --list

# 同一份任务清单跑了两遍（改代码前后、同一台机器）：逐批逐项是否相同
python experiments/s7/s7_compare.py experiments/s7/results/out_valid_b/baseline runs/s7/baseline
```

任务的 `root` 相对该 JSON 所在目录解析；未指定时使用项目当前代码。
