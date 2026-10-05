# experiments/s7：基线能不能让每块料独立测量（2026-10-05）

关键结论见 [实验总览](../../docs/EXPERIMENTS.md)，过程和完整表格见 [S7 完整记录](REPORT.md)；推后的问题在 [TODO.md](../../docs/TODO.md)。这是当前基线验证的实验入口。

| 文件 / 目录 | 是什么 |
|---|---|
| `s7_jobs.py` | 生成任务清单：基线（默认参数）× 四种布料 × 种子。`ideal` 是整条线读真值的对照，`creep15`、`creep075` 是给料带慢走减速的两档 |
| `s7_analyze.py` | 批量报告：导入共用的 [`singulator/audit.py`](../../singulator/audit.py)，统计逐块独立测量、失败原因及批次状态。旧完整结果可直接重算。口径见 [计量段设计](../../designs/station/DESIGN.md#独立测量验收口径) |
| `s7_analyze_checks.py` | 共用审计的自检（17 项，构造结果，不跑物理），包含撤销记录、JSON / CLI 摘要、主指标与后续检查分离及批量报告入口 |
| `s7_feed.py` | 给料机头做了什么：每次放料下了几块、两块过边隔多久、多久挡住 S1、一起下去的料后来怎样 |
| `jobs/jobs_*.json` | 从原 `_scratch/` 提出的 7 份任务清单；`root` 相对清单指向 `../snapshots/candidate` |
| `snapshots/candidate/` | S7 后期仿真源码（`f6794775`），共用审计收尾前的副本。早期运行使用过不同版本，以结果中的 `source_sha` 为准 |
| `results/out_valid_b/`、`results/out_valid_c/` | **结论用的批**：修正后重跑 7101–7130，另用新种子 7131–7160 扩大验证；当轮仿真源码的 `source_sha` 以 `f6794775` 开头 |
| `results/out_valid/` | 首轮验证（种子 7101–7130），修 S1 失效误报之前的代码（`5414afef`）：6 批被误报停线 |
| `results/out_small60/` | 小批定位，出口 0.60 m，种子 7001–7010（`5414afef`） |
| `results/out_small/` | 小批定位，出口 0.65 m（定 0.60 之前），同样的种子：`baseline`、`ideal`、`creep15`、`creep075` |
| `results/out_s6check/` | 用当轮仿真代码加 `S6_LINE` 重跑第 S6 节的 24 批，核对默认值改了之后旧结果还能复现 |
| `archive/changes_code_20261005.diff` | S7 仿真阶段的代码改动（`experiments/s6/snapshots/candidate3` → 当时项目根目录），不包含随后共用审计的收尾修改 |
| `archive/scratch/` | 原 `_scratch/` 中剩余的日志、核对脚本及自检记录。保留原样用于追溯；内部历史脚本路径未改写，不作为当前运行入口 |

直接重算已保存的结论（在项目根目录，不跑物理仿真）：

```powershell
python experiments/s7/s7_analyze.py experiments/s7/results/out_valid_b experiments/s7/results/out_valid_c --list
```

重新仿真的命令见 [S7 完整记录](REPORT.md) 末尾。早期任务清单保留原参数，但当前保留的 `candidate` 是后期版本，不能用它宣称精确复现早期代码结果。
