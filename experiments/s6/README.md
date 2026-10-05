# experiments/s6：第 S6 节的批量试验（2026-10-04）

结果和分析见 [S6 完整记录](REPORT.md)，当前结论见 [实验总览](../../docs/EXPERIMENTS.md)。这里是历史试验资料和仍由 S7 共用的跑批入口。这一节的线是 10-04 的默认值：出口 0.65 m、相机控制给料、计量等读数稳定；10-05 起默认值是基线（第 S7 节），`s6_jobs.py` 生成的清单会显式带上 `--lane-w .65 --feeder-sensing vision --weigh-model steady`。

| 文件 / 目录 | 是什么 |
|---|---|
| `s6_jobs.py` | 生成任务清单（配置 × 布料 × 种子）。`CONFIGS` 是每个配置的命令行，`SET` 是没有命令行参数、直接改模块常量的那几项，`S6_LINE` 是 10-04 的线；`--old-code` 配合 `--root` 用于下面三个旧快照（它们没有 `S6_LINE` 那三个参数） |
| `s6_run.py` | 多进程跑清单；已有结果的批跳过，中断后同一条命令续跑；输出目录里放一个名为 `STOP` 的文件就停。第 S7 节也用它 |
| `s6_analyze.py` | 汇总表：需人工批、作废成因、给料带动作、同批配对比较（`--screened` 去掉超数值筛查限的批，`--alias` 把重跑的配置并到原名下） |
| `s6_pairs.py` | 一起放下的两块料后来怎样：间距判据、同一对料在两种结构下的间距差 |
| `jobs/jobs_*.json` | 7 份历史任务清单；`root` 相对清单指向 `../snapshots/candidate*`。整理仅修正路径，任务参数保持不变 |
| `snapshots/candidate/` | 第一轮的代码快照（结果摘要里 `source_sha` 以 `142d60ca` 开头的批） |
| `snapshots/candidate2/` | 第二轮的代码快照（`f44ce8a5`）：加了跟踪血缘修复和计量段两条可选规则 |
| `snapshots/candidate3/` | 10-04 交付的代码（`664de0ca`）：再加"等排料板时继续看"。10-05 之前项目根目录就是它 |
| `results/out/` | 种子 7001–7030 的 23 个配置，「S6 结果」前九段各表的数从这里来；`base_v5` 是核对第一轮程序的 8 批 |
| `results/out_v6/` | 交付的代码上的重跑和对照：`final_v6`、`robust_v6`、`proto_a_v6`（各 118 批，与 `results/out/` 里同名配置逐项相同）；`drum_head_v6`（并齐 7011 一批）；`proto_a_mu30`、`proto_a_mu60`（钢面摩擦 0.30 / 0.60） |
| `results/out_holdout/` | 样本外：种子 7031–7060 的 `final`、`robust`、`proto_a`（交付的代码），种子 7016–7060 的 `before`（第一轮代码） |
| `archive/originals_before_20261004.zip` | 2026-10-04 改动之前的 7 个代码文件和 4 份文档 |
| `archive/changes_code_20261004.diff` | 10-04 的全部代码改动（改动前 → `candidate3`） |
| `archive/scratch/` | 原 `_scratch/`：日志、核对与测速试跑、分析草稿、传输打包。保留原样用于追溯；内部历史脚本路径未改写，不作为当前运行入口 |

每批两个文件：`<布料>_<种子>.json`（摘要）和 `<布料>_<种子>.result.json.gz`（完整结果，没有轨迹和模型 XML）。相贴 7028、7029、7041 的摘要里只有 `error`：布料生成把料摆进了设备，模型拒绝运行。

复现命令见 [S6 完整记录](REPORT.md) 末尾。
