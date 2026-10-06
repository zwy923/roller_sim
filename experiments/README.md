# 实验脚本

批量跑批和逐块审计的脚本都在这一个目录里。结论见 [实验总览](../docs/EXPERIMENTS.md)，每轮的完整记录见 [S8](../docs/history/EXPERIMENTS_S8.md)、[S7](../docs/history/EXPERIMENTS_S7.md)，指标定义见 [计量段设计](../designs/station/DESIGN.md#独立测量验收口径)。以前的版本靠 git 找，不在这里另存。

| 文件 | 是什么 |
|---|---|
| `jobs.py` | 生成任务清单：配置 × 四种布料 × 种子，每批 3 + 种子 % 3 块、时限 200 s。配置见文件开头：`baseline`（现在的线）、`ideal`（全线读真值的对照）、`beam_stop`（S7 的线），以及 S8 试过的 `creep15`、`stop_past`、`beam_high` |
| `run.py` | 多进程跑任务清单，中断后同一条命令续跑；`--traj` 另存每批的轨迹和模型，给 `replay.py` 用 |
| `analyze.py` | 逐块审计的批量报告：独立测量成功率、失败原因、批次状态。审计本身在 [`singulator/audit.py`](../singulator/audit.py) |
| `analyze_checks.py` | 审计的自检（构造结果，不跑物理），`checks/run_all.py` 会跑它 |
| `compare.py` | 几个配置在同一批任务上配对比较：独立测量、一次放下多块、点动、清线时间 |
| `feed.py` | 给料机头每次放料：下了几块、两块过边隔多久、多久挡住 S1、一起下去的料后来怎样 |
| `replay.py` | 回放 `run.py --traj` 存下的轨迹，估计换一个停带规则（光束挪位置、质心过边几毫米停）后给料带会早多少停、第二块还会不会过边 |
| `diff.py` | 同一份任务清单跑了两遍（改代码前后、同一台机器）：逐批逐项是否相同 |

在项目根目录运行：

```powershell
python experiments/jobs.py --seeds 7201-7260 --configs beam_stop,baseline --out runs/jobs.json
python experiments/run.py --jobs runs/jobs.json --out runs/batch --workers 10
python experiments/compare.py runs/batch --configs beam_stop,baseline --list
python experiments/analyze.py runs/batch --list
python experiments/feed.py runs/batch --config baseline
python experiments/diff.py runs/batch/baseline runs/batch_again/baseline
```

任务的 `root`（`jobs.py --root`）相对任务清单所在目录解析，指另一份代码；不写就用项目当前代码。

## 已保存的结果

新跑的批放 `runs/`。要长期留的放 `experiments/results/`。这两个目录都不进 git，只有跑它的那台机器上有，要留请另行备份。每批的摘要里记着它自己的命令行参数和源码哈希。

S7 的批原来在 `experiments/s7/results/`，在 Windows 上整个挪过来：`Move-Item experiments\s7\results experiments\results`。挪过来之后，各组如下：

| 组 | 是什么 |
|---|---|
| `out_valid_b/`、`out_valid_c/` | **S7 结论用的批**：修正后重跑 7101–7130，另用新种子 7131–7160 扩大验证；源码哈希以 `f6794775` 开头 |
| `out_valid/` | S7 首轮验证（种子 7101–7130），修 S1 失效误报之前的代码（`5414afef`）：6 批被误报停线 |
| `out_small60/` | S7 小批定位，出口 0.60 m，种子 7001–7010（`5414afef`） |
| `out_small/` | S7 小批定位，出口 0.65 m（定 0.60 之前），同样的种子：`baseline`、`ideal`、`creep15`、`creep075`（当时的配置名） |
| `out_s6check/` | 用 S7 的代码加 `S6_LINE` 重跑第 S6 节的 24 批，核对默认值改了之后旧结果还能复现 |

读已保存的 S7 结论，不重跑仿真：

```powershell
python experiments/analyze.py experiments/results/out_valid_b experiments/results/out_valid_c --list
```

S8 的 1080 批跑在云端会话的容器里（`runs/`），容器回收后就没有了；表格在 [S8 完整记录](../docs/history/EXPERIMENTS_S8.md)，末尾有复现命令。

S7 当时用的源码快照（`f6794775`）、任务清单和第 S6 节的脚本在 git 标签 `before-refactor` 上，取用方法见项目 [README](../README.md#2026-10-05-结构重构)。
