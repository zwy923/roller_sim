# experiments/s8：给料机头一次放下多块（2026-10-06）

关键结论见 [实验总览](../../docs/EXPERIMENTS.md)，过程和完整表格见 [S8 完整记录](REPORT.md)；问题来源是 [TODO.md](../../docs/TODO.md) 第 1 节。跑批和逐块审计沿用 S7 的脚本（`experiments/s7/s7_run.py`、`s7_analyze.py`、`s7_feed.py`）。

| 文件 | 是什么 |
|---|---|
| `s8_jobs.py` | 任务清单：和 S7 同样的批（四种布料 × 种子，3 + 种子 % 3 块，200 s），配置是基线和 TODO 第 1 节的三种改法 |
| `s8_beam.py` | 回放已保存的运行（`s7_run.py --traj` 存的 `model.xml` + `trajectory.npz`）：换一个停带规则（光束挪位置、质心过边几毫米停），给料带会早多少停、第二块还会不会过边。停带之前和原运行完全相同，所以停带时刻是精确的；停带之后是估计 |
| `s8_compare.py` | 几个配置在同一批任务上并排：独立测量、一次放下多块、点动、清线时间，逐批配对比较 |

原始结果放在 `runs/`（不进 git）。复现命令见 [S8 完整记录](REPORT.md) 末尾。
