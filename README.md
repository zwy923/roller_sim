# 煤矸逐块分选线：MuJoCo 三维筛查模型

给**试验样机**选机械结构、排查流程问题用的筛查模型，不是标定 DEM。用途：看清部件如何相互作用、找出失败形态、比较结构方案。

    给料带（高 10 cm，逐块放）→ 主带 0.40 m/s → 犁面 → 车道 0.60 m → 主带机头（下台阶 10 cm）
    → 缓冲带 1.5 m、0.8 m/s → 计量带 0.90 m（称重 + 测体积）→ 排料板（带侧挡板）→ 煤 / 矸石

- 给料带：[designs/feeder/DESIGN.md](designs/feeder/DESIGN.md)。
- 缓冲带、计量带、排料板和整条线的**流程逻辑**（每段读什么信号、做什么）：[designs/station/DESIGN.md](designs/station/DESIGN.md)。
- 排料板独立模型：[designs/flip_separator/README.md](designs/flip_separator/README.md)。
- 给料机头小试（真实滚筒、交接间距、台架方案）：[designs/transfer_trial/DESIGN.md](designs/transfer_trial/DESIGN.md)。
- 当前实验结论见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)，完整表格和历史过程由总览链接到各阶段记录；本文和 `designs/` 只讲结构、控制和用法。
- 实验脚本：[experiments/README.md](experiments/README.md)（跑批、审计、配对比较、回放，都在这一个目录）。

> 2026-09-30 起只有这一条线（用户：删主线，把完整结构当主线）。原主线（挡料门、斜置辊床、3× 卸料带）、给料带单独试验线、它们的实验记录和回放台都已删除；删除前的整个项目打包在上一级目录 `roller_sim_backup_20260930_before_single_line.zip`。
> 所有数字只用于同一模型内的结构比较，不是实物发生率、产率或选型依据。

## 样机的使用场景（用户，2026-09-20）

- 来料：**300–500 mm 粒级原煤 / 矸石，一批 3–5 块**，从溜槽 / 料斗放下。上游保证**解叠（单层）**，但**分布不受控**：横向位置和朝向任意，料可以互相贴着。
- 流程：逐块放料 → 单列 → 缓冲 → 逐块称重 + 测体积 → 按密度分选。
- 有摄像头看着整条线；一批走完才放下一批。

所以到达率由用户控制，排队论不是约束。要看的是：**每块料能不能单独上秤**（作废停机次数）、有没有错误放行或没测就落仓、单批走完的时间，以及造价。

2026-10-04 当前验证重点：**逐块独立测量及其整线验收口径**。称重 / 体积模型的进一步真实性、人工恢复与报警后滑行、实验存档机制暂不扩展；保留现有假设和诊断，不把这些后续问题作为本轮改进的前置条件。

2026-10-05（用户）：线固定为一套**基线**，结构扫描和节拍优化暂停，先回答"这套结构能不能让每块料独立测量"。基线 = 下面的「当前结构」（出口 0.60 m）+ 给料带控制器直接读每块料的真实质心 + 计量装置按固定 1 s 的概念装置处理。主指标是**独立测量成功块数 / 全部投入块数**；结果在 [EXPERIMENTS.md](docs/EXPERIMENTS.md) 第 S7 节，有意推后的问题（读数稳定核验、相机控制给料、节拍等）记在 [TODO.md](docs/TODO.md)。

2026-10-06（用户：修两处模型问题，处理机头一次放下多块）：给料机头改为**按质心停带**——头一块质心过机头 3 mm 就停，让料自己翻（`--feeder-stop centroid`，默认）。没见过的新种子上一次放下多块 10.8 % → 3.9 %，独立测成 97.0 % → 98.4 %；见 [EXPERIMENTS.md](docs/EXPERIMENTS.md) 第 S8 节。这条规则要毫米级地知道质心什么时候过边，基线读真值才做得到（[TODO.md](docs/TODO.md) 第 1 节）。

## 运行

```powershell
python plough.py                                          # 一批 5 块，出视频
python plough.py --seed 392 --duration 150 --no-video     # 指定种子、时限，不渲染
python plough.py --layout aligned --seed 6001 --count 4   # 并齐布料（还有 touching、flat、oblique）
python plough.py --bench --layout touching --seed 5001    # 只看给料机头：料全落到主带上就结束
python plough.py --scenario touching --seed 4004          # 验收：两块相贴一起上秤，必须停住
python checks/run_all.py                                  # 全部自检，1–3 分钟；--full 再加物理验收（再约 10–15 分钟）
python checks/plough_checks.py                            # 给料带、犁面、车道：几何、驱动、块体、撤离、放料控制（12 节）
python checks/station_checks.py                           # 缓冲 / 计量 / 排料板：几何、传感器、称重、控制、物理验收（约 15 分钟）
python checks/feeder_regression_checks.py                 # 放料判定（不跑物理）
python checks/geometry_regression_checks.py               # 带面支撑、设备间隙扫描
python checks/physics_regression_checks.py                # 动力学、有限制动、接触和诊断
python checks/material_checks.py                          # 煤矸组成、质量守恒
python checks/flip_separator_checks.py                    # 排料板独立模型：连杆、行程、通道间隙
python checks/architecture_checks.py                      # 包的分层：依赖方向、控制器不读真值、可调常量可被 --set 改到
python checks/timestep_checks.py                          # 同来料三档步长；不等同于实物标定
python designs/transfer_trial/sweep.py                    # 小试：机头几何 × 布料矩阵
python plough.py --set control.station.APPROACH=.65       # 临时改一个模块常量（可重复；python -m singulator.tuning 列出全部）
python experiments/jobs.py --seeds 7201-7260 --configs beam_stop,baseline --out runs/jobs.json   # 批量清单：配置 × 四种布料 × 种子
python experiments/run.py --jobs runs/jobs.json --out runs/batch --workers 10   # 多进程跑，中断后同一条命令续跑
python experiments/analyze.py runs/batch --list                                 # 逐块审计：独立测量成功率、失败原因
python experiments/compare.py runs/batch --configs beam_stop,baseline --list    # 同批配对比较
python experiments/diff.py runs/batch/baseline runs/batch_again/baseline        # 同一批任务跑两遍：逐批逐项是否相同
python experiments/analyze_checks.py                                            # 逐块审计的自检（不跑物理）
```

每次运行在 `runs/<名字>_<时间>/` 下保存：
- `result.json`：`measurement_audit`（独立测量主指标、逐块失败原因、批次状态及后续检查），配置、派生几何、结局、驱动载荷、撤离事件、逐块信息；`feeder`（每次放料、停带原因、点动）、`transfer`（机头每次放下了哪几块、挂边、下游间距）、`station`（每件料的称重、体积、密度、判定、作废原因、落仓，核验项）、`perception`（相机、光电、扫描）、`taken_off`（作废移出的料）；
- `trajectory.npz`：每 10 ms 一帧，各块位姿（pos + quat）、质心、前缘、接触类别位掩码、犁面角、犁面接触法向力、各驱动速度系数、称重读数、排料板角度；配合 `model.xml` 可精确重建每块料；
- `model.xml`；`video.mp4`：上半跟随斜视，下半正交俯视。

`runs/` 放新运行和临时试跑，不进 git。要长期留的批次结果放 `experiments/results/`（不进 git，只有跑它的机器上有；压缩完整结果不包含轨迹与模型 XML）。运行方式见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 当前结构

| 段 | 尺寸 | 作用 |
|---|---|---|
| **给料带** x −2.50 → −0.30 | 宽 1.20 m、长 2.20 m，顶面比主带高 **10 cm**；前送 0.10 m/s、慢走 0.03 m/s；落料光束 S1 在机头前 0.10 m | 整批暂存；质心过机头边缘的料自己翻下去，一次一块；按预测放下一块（控制器读每块料的真实质心；头一块质心过边 3 mm 就停带，S1 只做诊断） |
| **主带** x −0.30 → 2.24 | 宽 1.20 m，0.40 m/s（速度给定的保持位平板） | 送料，一直通到车道出口后的机头 |
| **犁面** x 0.50 → 折弯 (1.29, 0.65) | 角度沿 y 从 55° 转到 20°（24 段折线），9 根 Ø90 自由竖辊，高 0.50 m，底缝 5 mm；最底下一根的下切线与车道外壁齐平 | 把料扫向低侧并拉出前后错位，错位预算 0.45 m |
| **低侧侧带** x 0.37 → 2.24 | 1.5× = 0.60 m/s，覆盖整个漏斗和车道 | 给不出拱脚需要的向后反力；差速拆并排 |
| **车道** | 净宽 **0.60 m**（用户 2026-10-05；此前 0.65 m），长 0.90 m；外壁从 x 1.375 起（让开底辊的撤离行程） | 单列闸口 |
| **缓冲带** x 2.24 起 | 宽 0.70 m，长 **1.5 m**，**0.8 m/s**，顶面比主带低 **10 cm**；S2（入口后 0.65 m、带面上 1.5 cm）、S3（停位） | 比主带快，把一起翻下来的料拉开；计量带忙时把头一件停在 S3 |
| **计量带** | 宽 0.70 m，长 0.90 m，0.40 m/s，与缓冲带平接；带和挡边坐在称重传感器上；上方三头深度扫描 | 停一次，同时称重和测体积：停稳后固定 1 s 出数（概念装置，不核验读数稳定） |
| **排料板** | 板宽 0.70 m，7 根纵筋，衬 UHMW-PE，**两侧挡板 0.25 m** 随板翻；S4 在板入口下 | 密度 ≥ 1800 kg/m³（占位）抬板落矸，否则落板走煤 |
| **挡墙** | 挡边 / 侧带 / 车道外壁 / 竖辊统一 0.50 m | 料平躺时最高就是短轴，本粒级上限 0.50 m |
| **外挡边** | 止于 x 0.55，距犁面扫掠包络 10 mm | 留出犁面摆动窗口；撤离期间这段带边是敞开的 |

驱动件：给料带、主带、侧带、缓冲带、计量带、犁面撤离执行器、排料板油缸；犁面竖辊不驱动。
传感器：给料带相机、单列段相机、计量段相机、排料板相机，光电 S1–S4，三头体积扫描（实体在 `singulator/machine/sensors.py`，信号模型在 `singulator/sensing/`），控制器只读它们给出的信号（`--sensing vision`，默认）。两处例外是 2026-10-05 的基线定的：给料带控制器读每块料的真实质心和轮廓（`--feeder-sensing oracle`，默认；2026-10-06 起它按质心停带，S1 只做诊断），计量装置停稳后固定 1 s 出数（`--weigh-model fixed`，默认）。

2026-10-04 代码状态：
- 相贴料团按前缘预送、慢走，避免用整团形心预送时把前一块提前推出机头；孤立单块仍按轮廓形心。（对照选项 `--blob-lead centroid` 已于 10-05 删除。）
- 跟踪器合并时不再把短时未匹配的旧轨迹重复计为新料；料团分开后保留来源关系。
- 测量有效、等待翻板期间继续检查邻料和多块；有新料靠近会撤销放行、作废停住。
- `--weigh-stop centre`（居中停）和 `--buffer-approach slow`（接缝前最后 0.45 m 按计量带速度运行）已经实现，**默认仍是 `rear` / `full`**。称重判稳时限默认仍为 3 s；S6 的 `robust` 等实验组合不是默认配置。

2026-10-05 基线（用户定的三项，默认值随之改了；上面 10-04 的几条仍然成立）：
- **出口 0.60 m**（`--lane-w`，此前 0.65）：折弯和主带机头顺流后移 7 cm，犁面多一根竖辊；单列段相机的高度和门架立柱改成随出口宽度算（0.60 m 时相机高 2.50 m）。
- **给料读质心**（`--feeder-sensing oracle`）：给料带控制器不经过相机，直接读每块料的真实质心和轮廓；没有"料团"，慢走和放出都按质心判。相机控制给料（料团按前缘预送等）留在 `--feeder-sensing vision`。计量段、犁面仍读模拟相机和光电；`--sensing oracle` 是整条线都读真值的对照。
- **计量固定 1 s**（`--weigh-model fixed`）：计量带停稳后 1 s 出数，取最后 0.5 s 的平均；读数稳不稳只记录（`steady`），不作废。原来"3 s 内读数稳定才出数、否则作废"的规则留在 `--weigh-model steady`。
- 同日修了读质心路径上 S1 失效的误报（见文末「2026-10-04、10-05 修复」）。
- 第 S6 节的批次是 10-04 的线。用现在的代码复现要带 `--lane-w .65 --feeder-sensing vision --weigh-model steady`（相机路径上总是光束停带，不受下面 10-06 的改动影响）。

2026-10-06 基线（S8；上面几条仍然成立）：
- **按质心停带**（`--feeder-stop centroid`）：给料带不再慢走到料翻下去挡住 S1，头一块质心过机头 3 mm（`control.feeder.STOP_PAST`）就停，料自己翻；挂边的料每次点动 5 mm（`JOG_STEP`）、再等 0.5 s。S1 在这条路径上只做诊断。原规则留在 `--feeder-stop beam`。
- **早挡慢走**（`control.feeder.EARLY_BEAM_CREEP`）：本次放料还没有质心过边光束就被挡，慢走到有质心过边再停，不再反复起停。
- 第 S7 节的批次是 10-05 的线。用现在的代码复现要带 `--feeder-stop beam --set control.feeder.EARLY_BEAM_CREEP=0`（`experiments/jobs.py` 的 `beam_stop`）。

**撤离**：犁面整体绕折弯端铰点铰接，撤离 −25°，相机判卡触发。复位要同时满足：区段走空；没有任何料的整块轮廓伸进犁面回位要扫过的区域（留 10 mm）。复位过程中每 10 ms 重查，有料进入就停在原位、退回保持。20 s（`--face-hold-max-s`）内得不到复位许可就停线（`jammed` / `retract_hold_timeout`），不带料强收；开始复位后 6 s 未到位记 `face_fault`。

坐标系：x 沿带流向，y 从低侧基准边（`lane_y` = 0.05）指向带另一侧，z 向上，主带面 z = 0。

## 料的模型

`singulator/lumps.py`：五族加权凸包（等轴 30 / 块状 35 / 板状 20 / 柱状 10 / 刃状 5 %），摩擦按接触类别固定。形状比例没有可核验的来源，均为筛查假设。

材料：默认生成煤为主、夹矸煤、矸石为主三类，每块先生成煤 / 矸端元干质量占比，再由组成算等效密度，用实际凸包体积计算质量与重力；详见 [MATERIAL_MODEL.md](docs/MATERIAL_MODEL.md)。默认按块数概率 45 % / 25 % / 30 %，属于待标定假设。

| 参数 | 值 | 口径（全部待标定） |
|---|---|---|
| 煤端元有效密度 | 1250–1450 kg/m³ | 不是整块混合密度 |
| 矸石端元有效密度 | 2250–2700 kg/m³ | 不是堆积密度 |
| 块–胶带 | 0.55 | 煤 / 岩对橡胶带面 |
| 块–钢板 | 0.45（衬 UHMW-PE 取 0.20） | 煤对普通钢 |
| 块–块 | 0.60 | 煤对煤 / 岩对岩 |
| 接触阻尼比 | 1 | 软接触参数，不等于实测恢复系数；不模拟破碎 |

默认 condim 3（滑动摩擦）；`--contact-condim 4/6` 和 `--torsional-friction` / `--rolling-friction` 可启用扭转 / 滚动摩擦（系数单位是米）。

**粒度口径**：方孔筛约束的是中间轴，所以 `--size-min/--size-max` 是筛孔，长轴 = 筛孔 / b；`--size-long-max`（0.50 m）截住长轴，代表上游挑矸 / 破碎。这个上限截的是 body x 轴向范围，不是最大卡尺尺寸：约 6 % 的料平躺俯视跨度超过 0.55 m，最大约 0.67 m。

**布料** `--layout`：
- `scatter`（默认）：单层、任意分布，在 `--feed-len` 料区内随机抽 x / y / 偏角，只要求俯视凸包不重叠（允许贴着）；
- `aligned`（并齐）、`touching`（相贴）、`flat`（扁平）、`oblique`（斜放）：给料机头小试的布料，见 [designs/transfer_trial/DESIGN.md](designs/transfer_trial/DESIGN.md)。

## 结局判据

**优先看 `measurement_audit`。** 主指标是独立测量成功块数 / 全部投入块数；有效且未撤销、真值单块、无秤外接触、扫描有效才计成功。读数稳定、质量误差和正确落仓单列后续检查，不进入这个成功率。作废移除仍留在分母。完整定义见 [独立测量验收口径](designs/station/DESIGN.md#独立测量验收口径)。

2026-10-05 已接入：`singulator/audit.py` 提供唯一审计实现，普通 `result.json` 写入 `measurement_audit`（`counts`、`verification`、`later_checks`、`batch`、逐块 `lumps`）；命令行先显示独立测量摘要；S7 分析复用同一函数。`counts.success_rate` 为 0–1 的比值，`batch.status` 是包含后续检查的总体状态，不能用它代替主指标。

旧字段保留为过程诊断：`single_file`、`single_file_after_unjam`、出口间距和 `first_pass_yield` 只描述主带机头通过；后者是尾缘通过块数 / 投入块数。出口并排后被缓冲带拆开可以测成；单列通过后也可能作废。旧 `station.verification` 仍含质量误差检查，单列显示，不改变新主指标的定义。

- **每件料**（`station.items`）：称得的质量、体积、密度、分选去向、作废原因；核验项 `station.verification`：
  - `false_valid`：判为有效、但真值不是"一块料、不碰秤外、称重误差 ≤ 2 %"的件——必须为空；
  - `landed_unmeasured`：落进料仓却没有一件有效的单块测量对应它——必须为空；
  - `false_void`：真值是好件却作废了——代价，不是错误。
- `counts.holds`：作废停机次数（样机上每次要人工分开、重测）；`taken_off`：随作废件移出的料；`outcome.line_clear_s`：最后一块落仓的时刻。
- `outcome.classification`：`single_file` / `single_file_after_unjam` / `abreast_at_cut`（两块同时跨过主带机头）/ `dropped` / `incomplete`（时限到还有料在线上）/ `jammed`（`retract_hold_timeout`、`unjam_exhausted`）/ `face_fault` / `station_fault`（排料板不到位或路径超时、光电误挡 / 长时遮挡 / 失效、视觉丢失 5 s）。
- 卡堵：单块料在区段（犁面起点前 0.3 m 到车道出口）内连续待满 `--jam-window`（4 s），相机看到的质心前进 < `--jam-speed` × 窗口，且窗口内没有料的尾缘离开车道，即判停滞，触发撤离。还在给料带上排队、或被计量段按住的料不算。
- `drives.*` 同时给净载荷（带符号 50 ms 平均）和单向载荷（先整流再平均）；**50 ms 平均只是模型诊断量，不是设计载荷**。驱动没有电流、发热和保护跳闸模型。
- `numerics`：穿透、MuJoCo 警告、求解迭代上限与异常；任一超限都使 `ok=false`。穿透只筛设备上的接触：料离线后落到地面（代表料仓）或落在另一块已离线的料上，记在 `numerics.landed`，不算超限（2026-10-06 起）。`ok=true` 不代表收敛或现实准确。

## 已知问题

- **撤离够不着折弯口和车道**：铰点在折弯处，铰点附近几乎不动，车道外壁不动；折弯口和车道里的拱撤离解不开。
- **保持超时从退到位起计时**，不看料是否恢复了前进。
- **撤离期间外侧带边敞开**：外挡边止于 x 0.55，犁面退开时料可能从那里掉下（`dropped`）。
- **犁面是有限力矩动力学执行器**：托架质量、伺服刚度阻尼是假设。
- **驱动无电流、发热和跳闸模型**；侧带 15 kN 限力是假设。
- **接触参数尚未证明收敛**：默认 dt 0.25 ms、solref 2 ms；三档步长对照里部分工况清空时间仍敏感。
- **料不可破碎，恢复系数未标定**；密度、摩擦、块形均是假设。
- **给料机头默认是尖边**：实物是滚筒，两条带之间会有交接间距；滚筒、尾辊和交接间距已可建模（`--feeder-head-d` 等），哪种几何能用要先做台架小试。
- **质心纵向齐平、并排翻下的两块**，缓冲带速差拉不开（它们同时落到快带上），只能作废停机、人工分开。基线下没测成的料几乎全部出自给料机头一次放下了不止一块；按质心停带之后剩下的主要是质心真的齐平的和被头一块拖过边的（第 S8 节）；拆不拆得开看两块过主带机头时错开多少（[S6 料块间距分析](docs/history/EXPERIMENTS_S6.md#三一对料拆不拆得开看过主带机头时错开多少)）。
- **有意推后的问题**（机头一次放两块、读数稳定核验、相机控制给料、圆料在接缝处、节拍、模型本身的缺陷）集中记在 [TODO.md](docs/TODO.md)，每条写了现在怎么简化的、已经知道什么、什么时候该捡起来。
- **停车不是无限制动**：给料带、缓冲带、计量带用有限制动力；报警后结束模拟，不模拟随后的滑行。
- **相机、扫描、称重的模型偏乐观**（凸料、无遮挡、理想力传感），见 [designs/station/DESIGN.md](designs/station/DESIGN.md)「实物风险」。

## 代码结构

2026-10-05 起 `singulator/` 按依赖方向分层：**下面的层只引用上面的层**，信息单向流动——真值 → 传感器 → 控制器 → 驱动目标。分层说明、每 10 ms 采样内的先后顺序、怎样替换称重 / 体积装置，见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

```
README.md              总入口：当前结构、运行方法、目录导航
requirements.txt       Python 依赖；本地环境在 .venv/
docs/                  项目说明与实验结论
  ARCHITECTURE.md      代码分层、数据流、替换计量装置的方法、2026-10-05 删掉的选项
  EXPERIMENTS.md       当前实验结论、明细导航和常用命令
  history/             各轮实验的完整记录：S1–S5、S6、S7、S8
  TODO.md              有意推后的问题（2026-10-05 起）
  MATERIAL_MODEL.md    材料与密度模型
  REALISM_REVIEW.md    现实性核查及边界
plough.py              命令行入口：解析参数 → simulate.run() → 打印摘要
singulator/            模型包
  config.py            全部参数的唯一来源：PARAMS 表 → 命令行、default_config()、validate()
  tuning.py            --set MODULE.NAME=VALUE：一次运行内临时改模块常量，并记入 result.json
  lumps.py             形状族、组成 / 密度抽样、质量
  geom2d.py  series.py 平面几何（凸包、重叠、间距）；载荷序列统计
  audit.py             共用逐块审计：独立测量成功率、失败原因、批次状态和后续检查
  simulate.py          run(cfg)：建线 → 循环 → 出结果
  machine/             设备本体：纯几何 + MJCF，不动、不判断
    feed_belt.py  plough.py  station.py  separator.py  sensors.py   各段自己的尺寸与零件
    layout.py          derive(cfg)：各段依次摆放，得到全包共用的几何字典 d
    assembly.py        build_xml()：按固定顺序拼装模型；接触类别；活动件全行程间隙扫描
  physics/             被仿真的对象：drives.py 各条带的虚拟电机，actuators.py 犁面伺服与排料板，
                       lumps.py 料块真值（每 10 ms），numerics.py 数值筛查
  sensing/             控制器能知道的：vision.py 相机，beams.py 光电，volume.py 体积扫描，
                       weigher.py 称重（LoadCell + Weigher 接口），suite.py 汇总与故障注入
  control/             控制器，只读传感器信号：feeder.py 给料带，face.py 犁面撤离，station.py 缓冲 /
                       计量 / 排料板，supervisor.py 线级规则（联锁、判卡、停线）
  verify/              与真值对照，不回馈控制：station.py（false_valid 等核验项），feeder.py（每次放料
                       真正过边的料），transfer.py（机头交接记录）
  sim/                 一批料的运行：layouts.py 布料，scenarios.py 验收场景与作废件移出，line.py 整条线
                       的组装与每步 / 每采样的顺序，record.py 真值记录与轨迹，results.py 结果组装，video.py
checks/                自检（见「运行」）；run_all.py 一次跑完
designs/
  feeder/              给料带的设计说明
  station/             缓冲带、计量带、排料板与流程逻辑
  transfer_trial/      给料机头小试的台架方案（DESIGN.md）与仿真矩阵（sweep.py）
  flip_separator/      排料板独立模型（model.py：渲染、交互窗口、检查输出）
experiments/           批量实验脚本（见 experiments/README.md）：jobs.py（任务清单）、run.py（多进程跑批、可续跑）、
                       analyze.py（逐块审计的批量报告）、analyze_checks.py（审计自检）、compare.py（几个配置同批
                       配对比较）、feed.py（给料机头每次放料）、replay.py（回放轨迹，估计换停带规则的效果）、
                       diff.py（同一批跑两遍是否逐项相同）
  results/             要长期留的批次结果，保留 out* 组名（不进 git）
runs/                  新运行产物（不进 git）
```

第 S6 节（10-04 的结构与控制对比）只留下记录 [docs/history/EXPERIMENTS_S6.md](docs/history/EXPERIMENTS_S6.md)；它的原始结果 2026-10-06 已删除。重构前的源码、S6 / S7 的脚本、源码快照和任务清单在 git 标签 `before-refactor` 上，见下面「2026-10-05 结构重构」。

设备之间在仿真里不碰撞，活动件间隙只能靠 `motion_clearance`（`checks/plough_checks.py` 第 9 节、`checks/station_checks.py`）。

## 现实性边界

物性是未标定且无已核验手册来源的假设；形状是合成凸包（无凹面、不破碎、无水分粘附）；接触为软约束；驱动限力与反射质量是筛查假设。称重是接触力之和、体积是理想值，传感器信号（相机、光电、扫描是否可用）是占位参数的模型，没有用真实煤矸标定。结果只能用于明确物性、控制、数值敏感性边界下的机制筛查，不能当实物发生率、产率或选型依据。模型核查记录见 [REALISM_REVIEW.md](docs/REALISM_REVIEW.md)。

## 2026-10-05 结构重构

只动代码结构，不改机器和控制规则：同一环境下重构前后逐位比对（`result.json` 与轨迹），结果见 [ARCHITECTURE.md](docs/ARCHITECTURE.md) 末尾。用的时候要知道的几件事：

- **删掉的选项**（都是对照用的旧实现）：`--face-drive-model kinematic`、`--no-drive-limit`、`--face-shape straight` / `--skew-deg`、`--face-hinge upstream`、`--no-unjam`（改用 `--unjam-max 0`）、`--feed-band plough`、`--layout rows`、`--material-model legacy_binary`、`--blob-lead centroid`。带这些参数的旧命令会报"无此参数"；要复现当时的结果，用重构前的代码（下一条）。
- **重构前的代码和资料在 git 标签 `before-refactor`**（提交 `77c8907`）：当时的源码，第 S6 节的全部脚本，S6 / S7 的四份源码快照（`experiments/*/snapshots/`）、已保存的任务清单（`jobs/`）、过程记录（`archive/`）和 10-05 的文件整理记录。原始结果不在 git 里：S7 的在 `experiments/results/`（原 `experiments/s7/results/`），S6 的已删除。2026-10-06 起这些不再放在工作目录里。要用：先提交手头的改动，`git checkout before-refactor`，按那一版自己的 README 运行（`.venv` 和 `results/` 不受切换影响），用完 `git checkout main` 回来。
- **改模块常量**不再在任务文件里写 `singulator.station.X`，改用 `--set control.station.X=值`（任务文件的 `"set"` 同理）；`python -m singulator.tuning` 列出全部可改的常量及其所在模块。
- **换称重 / 体积装置**：`sensing/weigher.py` 的 `Weigher`、`sensing/volume.py` 的 `Scanner` 是工位控制器用到的全部接口，换实现不用动控制器。

## 2026-10-06 提速

单批运行快约 2.2 倍（同机 4 进程跑 14 批：总耗时 1224 → 558 s，单批 1.9–2.5 倍），结果逐位不变：只减少了每步、每帧 Python 里重复的计算和 numpy 调用，物理和控制规则没动。改了什么、怎么验证的、以后改这些路径要注意什么，见 [ARCHITECTURE.md](docs/ARCHITECTURE.md#2026-10-06-提速结果逐位不变)。

## 2026-10-06 修复

S8 的准备和试验里修的，自检在 `checks/physics_regression_checks.py`、`checks/feeder_regression_checks.py`。

| 项 | 修复前 | 修复后 |
|---|---|---|
| 相贴布料重叠 | 首尾相贴的料只往前面那一块上靠，不管并排的另一块：种子 7001–9000 的相贴布料 2000 批里 62 批初始重叠超过 1 mm（55 批料插进料，最深 11.5 cm；7 批伸进低侧挡边），线拒绝运行。S6、S7 因此少跑 6 批（原记录说"摆进设备"，不准） | 只在真有重叠时挪：横移回带宽内，或退到所有料后面再往前靠到 3 mm。没重叠的批布料逐位不变；8000 批修后初始重叠全为 0。拒绝运行时报出哪两个几何体重叠 |
| 数值筛查把落仓算进去 | 所有接触取最大穿透：料从排料板落到地面（代表料仓）那一瞬间常到 3–6 mm，超过 5 mm 就 `ok=false`（S6：46 批超限里 41 批只是落仓；S8 的 1080 批里 9 批） | 穿透只筛设备上的接触；和地面、两块离线料之间的接触记在 `numerics.landed`，不算超限。仿真本身不变 |
| 光束挡着时反复起停（读质心的路径） | 本次放料还没有质心过边、光束就被一块慢慢垂下机头的料挡住：停带，下一个采样又起放料，再被停（相贴 7125：2.8 s 里 53 次空放料） | 慢走到有质心过边再停（`EARLY_BEAM_CREEP`） |

## 2026-10-04、10-05 修复

前三条是读代码、跑种子 392 时发现的，中间五条是第 S6 节的试验和它的准备工作里查出来的，最后一条是第 S7 节验证时查出来的；每条都有自检（`checks/feeder_regression_checks.py`，`checks/station_checks.py` 的 `Tracker` 和 `Control`）。

| 项 | 修复前 | 修复后 |
|---|---|---|
| 光束挡着时又起放料（相机路径） | 料翻过机头时先挡住 S1，轮廓形心还没过机头 5 cm，相机仍把它算作排队的头一块：停带后马上又起一次放料，下一帧又被光束停住，反复空放（种子 392：0.07 s 内 4 次） | S1 挡着时不起放料。读质心的路径没有这一条，见 [TODO.md](docs/TODO.md) 第 1 节 |
| 放料报告（相机路径） | 控制器没有每块料的身份，`never_released` 把每块都列成没放出，`after_stop` 恒为 0 | 取核验记录：哪几块放出，哪几块是停带后才下去的 |
| 停带距离记录 | 给料带没停稳就被预送或下一次放料重新起动时，记录一直开着，把后面整段运行都算成停带行程（种子 392：0.756 m、8.16 s） | 没停稳就再起动的停带单列（`stops_interrupted`），不进最大值 |
| 预送把贴着的料带过机头（相机路径） | 贴在一起的料在相机里是一团。给料带按这一团的轮廓形心预送和慢走，而前面那块自己的形心比它靠前 0.2–0.5 m：预送途中前面那块已经翻下去，这次下料没有经过放料判定 | 一团料按前缘预送和慢走（原来的做法曾留作对照选项 `--blob-lead centroid`，10-05 删除）；一团料只是块数拿不准时不再暂停放料 |
| 跟踪记数 | 一团料散开后留下的旧轨迹（已不对应任何对象）在下一次合并时又把块数加一遍。一块料挨着别的料躺在给料带上，19 s 内被记成 13 块（随机 7001），到计量带判"多块"作废 | 这种轨迹合并时不加块数（`sensing.vision.GHOST_S`） |
| 跟踪血缘 | 两块前后相贴的一团分开时，这一团的轮廓形心在两块中间，离两块各自的形心都超过 0.20 m 的匹配门限，两块都成了没有来历的新对象；计量段发现原来的对象没了、原地多出一个不认识的，判"跟踪丢失"作废（相贴 7004） | 出现在刚失配轨迹的轮廓里的新对象继承它的血缘（`sensing.vision.SPLIT_PAD`） |
| 判好的件等排料板时没人看 | 密度算出之后、排料板到位之前（最长约 2 s），计量带上的料不再做隔离检查。停在 S3 的圆料滚过接缝贴上它，两块一起排出，后一块没测就落了仓（滚筒机头试验，并齐 7011） | 等排料板期间照旧检查"5 cm 内有别的对象、不止一个对象"，出现就作废停住 |
| 凸包计算 | scipy 的 ConvexHull 在 Windows 上每次调用都开一个临时文件；每仿真秒约 600 次，占一次运行的大部分时间，并行进程互相等 | 纯 Python 单调链（`geom2d.hull2`），2800 组点逐点相同，批结果相同 |
| S1 失效的误报（读质心的路径，10-05） | 放出的料 1 s 内没挡住 S1 就记一次漏检，连着两次判"光束失效"停线。给料读质心之后，计时从质心过机头边缘就开始，约十二分之一的放料翻得比 1 s 慢——光束随后都看到了。首轮验证 118 批里 6 批因此停线，16 块料没到秤上（第 S7 节） | 和相机路径一样：这次放料的料整块越过了光束、而光束从这次放料起一直没挡过，才算一次；连着两次才判失效。1 s 的兜底停带不变 |

## 2026-09-22 修复

按 2026-09-22 物理核查（核查文件已移除，要点如下）：

| 项 | 修复前 | 修复后 |
|---|---|---|
| 块体轮廓 | 编译后网格顶点按 body 位姿变换（seed 81 最大偏 0.18 m） | 按 geom 位姿变换（`physics.lumps.world_vertices`），与生成几何一致到 1e-8 m |
| 设备干涉 | 外挡边贯穿摆动犁面（静止时首根竖辊插入 35 mm，摆动中最深 52 mm） | 外挡边止于犁面扫掠包络前 10 mm；`motion_clearance` 全行程扫掠检查 |
| 撤离超时 | 20 s 后带料强收 | 保持退让、停机，记 `jammed` / `retract_hold_timeout` |
| 复位许可 | 区段判空时有豁免 | 区段判空不豁免，另加整块轮廓对回位扫掠区的检查，复位中持续检查 |
| 复位超时 | 超时未回零也当作待机 | 实际回零才待机；否则记 `face_fault` 并停机 |
| 通过判据 | 前缘碰到测量面即算通过 | 前缘 / 尾缘分开，走完 = 尾缘通过；掉料单独分类 |
| 停滞判据 | 前缘净位移、整批取最大值 | 单块质心位移 + 区段持续占用 + 车道出口通过；计划等待单列 |
| 载荷统计 | 带符号载荷取 p99；后来又先平均再分正负（交替载荷被抵消为零） | 净载荷平均与单向（先整流后平均）两类并列；50 ms 值只作诊断 |
| 犁面「力」 | 字段名 `face_loads`，易被当作接触力 / 缸推力 | 改名 `face_hinge_moment_over_chord` 并注明含义；另记接触对法向力与接触时间 |
| 接触分类 | 车道外壁落在 `other` | 新增 `lane_wall`、`side_belt` |
| 轨迹 | 只存 t 与前缘 | 存完整位姿、质心、接触位掩码、活动件状态 |
