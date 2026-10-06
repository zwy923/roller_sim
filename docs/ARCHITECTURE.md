# 代码结构（2026-10-05 重构后）

这一页只讲代码怎么组织。机器结构和控制规则见 [README](../README.md) 与 `designs/`。

## 分层

`singulator/` 按依赖方向分层，**每层只引用排在它前面的层**（`checks/architecture_checks.py` 会检查）：

| 层 | 放什么 | 约束 |
|---|---|---|
| 基础：`config` `tuning` `lumps` `geom2d` `series` `audit` | 参数表、可调常量、料块形状与材料、平面几何、载荷统计、逐块审计 | 互相之间基本独立 |
| `machine/` | 设备本体：每一段的尺寸和 MJCF 零件；`layout.derive(cfg)` 摆放，`assembly.build_xml()` 拼装 | 纯几何，不依赖 MuJoCo 运行时；不动、不判断 |
| `physics/` | 被仿真的对象：各条带的虚拟电机、犁面伺服、排料板、料块真值、数值筛查 | 只执行目标值，不做决定 |
| `sensing/` | 控制器能知道的东西：相机、光电、体积扫描、称重的模型 | 真值只在这里变成信号 |
| `control/` | 控制器：给料带、犁面撤离、工位，以及把它们连起来的 `supervisor` | **不引用 `physics/`、MuJoCo，不读真值** |
| `verify/` | 把控制器的判定和真值对照，写核验项 | 只读不回馈 |
| `sim/` + `simulate.py` | 一批料的运行：布料、验收场景、整条线的组装与步进、记录、结果 | 可以引用所有层 |

信息单向流动：**真值 → 传感器 → 控制器 → 驱动目标**。两处例外是仿真自己的"手"（`sim/scenarios.py` 的验收场景和作废件移出），它直接改料块位姿，并告诉工位拿走了哪几件。

## 一次运行

`simulate.run(cfg)` 建一个 `sim.line.Line`，循环 `advance()`，最后 `sim.results.assemble()`。

- **每个物理步**（`Line._step`）：`Supervisor.drive_targets()` 汇总各控制器的目标 → 写给各条带、排料板、犁面伺服 → `mj_step` → 电机按这一步的载荷响应。
- **每 10 ms**（`Line._sample`，顺序就是代码里的顺序）：
  1. 记录真值（`sim/record.py`：逐块观测、同时过线、漏斗占用）；
  2. 传感器成像（`Sensors.frame`：相机、光电、称重采样）；
  3. 犁面复位许可（`Supervisor.see_section`）；
  4. 工位（`Station.observe`），之后仿真移走作废件、执行验收场景；
  5. 给料带（`Supervisor.feed`：联锁 + `Feeder.observe`）；
  6. 写轨迹；
  7. 判卡：撤离或停线（`Supervisor.judge`）。

## 替换称重或体积装置

工位控制器只通过两个接口用计量装置，换实现不用改 `control/station.py`。

**称重**（`sensing/weigher.py`）。`LoadCell.force()` 是信号（现在是称重框上接触力之和）；`Weigher` 是装置。工位用到的全部方法：

| 方法 | 含义 |
|---|---|
| `track_zero()` / `clear()` | 空秤时跟踪零点 / 人工清秤后零点复位 |
| `begin() -> tare` | 一件料开始停稳，开始一次称重，返回这次的零点 |
| `zero_ok(tare)` | 开始称重时秤是不是空的（不是则判 `tare` 作废） |
| `read(t, rest, tare)` | 停稳后每 10 ms 问一次：没出数返回 `None`；出数返回 `mass_kg` 等字段，带 `void='unsteady'` 表示有数但不可用 |
| `describe()` / `limits()` | 写进 `result.json` 的说明 |

仿真每个物理步调 `load_step(Fz)`、每 10 ms 调 `sample(Fz)` 喂数。现有两个占位实现：`FixedTimeWeigher`（`--weigh-model fixed`，默认：停稳 1 s 出数）和 `SteadyWeigher`（读数稳定才出数）。接真实皮带秤的模型：继承 `Weigher`，重写 `verdict()`（或整个 `read()`），在 `WEIGHERS` 里登记，并把名字加进 `config.py` 里 `weigh_model` 的可选值。

**体积**（`sensing/volume.py`）。工位只调 `read(t, rest)`（停稳后每 10 ms 问一次，没结果返回 `None`，有结果返回一次 `dict(valid, reasons, volume_m3, ...)`）和 `describe()`。现在的 `Scanner` 是占位：体积取真实凸包体积，只模拟"这次扫描能不能用"。体积检测方案定了以后，最省事的接法是继承 `Scanner`、重写 `scan(t, lumps, volume, dist, belt)`——输入是扫描区内各块料的真实顶点，输出是装置给出的体积和是否可用；计时（`scan_s`）、记录（`scans`）、故障注入（`--fault scan_fail:N`）都沿用。然后在 `sensing/suite.py` 的 `Sensors.__init__` 里把 `self.scanner` 换成新类。

核验（真值对照）在 `verify/station.py`，跟装置实现无关，换装置后 `false_valid` 等核验项照常统计。

## 参数与可调常量

- **参数**只在 `config.py` 的 `PARAMS` 表里定义一次，命令行、`default_config(**over)`、`validate()` 都从它生成。代码里一律 `cfg['name']`，没有第二套默认值；缺键或拼错会直接报错。
- **可调常量**（各模块顶部的大写常量，约 120 个，`python -m singulator.tuning` 列出）不是参数。临时改用 `--set 模块.常量=值`，例如 `--set control.station.APPROACH=.65`，会记入 `result.json` 的 `config.set`。跨模块引用常量要写 `模块.常量`，不能 `from 模块 import 常量`（否则 `--set` 改不到，结构检查会报）。

## 2026-10-05 删掉和搬走的东西

删掉的命令行选项见 [README](../README.md#2026-10-05-结构重构)。旧模块到新位置：

| 重构前 | 重构后 |
|---|---|
| `machine.py` `assembly.py` `devices.py` `separator.py` | `machine/` 下按设备分开 |
| `drives.py`；`face.py` 的伺服部分；`tracking.py`；`physics.py` 的诊断 | `physics/drives.py` `actuators.py` `lumps.py` `numerics.py` |
| `perception.py` | `sensing/vision.py` `beams.py` `volume.py`；平面几何进 `geom2d.py` |
| `station.py` | 几何 → `machine/station.py`；控制 → `control/station.py`；称重 → `sensing/weigher.py`；真值核验 → `verify/station.py` |
| `feeder.py` | 几何 → `machine/feed_belt.py`；控制 → `control/feeder.py`（两套实现合一）；放料真值 → `verify/feeder.py` |
| `face.py` 的状态机 | `control/face.py` |
| `line.py` | 传感器汇总 → `sensing/suite.py`；场景与移出 → `sim/scenarios.py`；视觉判卡 → `control/supervisor.py` |
| `trial.py` | 布料 → `sim/layouts.py`；交接记录 → `verify/transfer.py` |
| `simulate.py` 的 `run()`（243 行） | `simulate.py`（循环）+ `sim/line.py` + `control/supervisor.py` + `sim/record.py` + `sim/results.py` |

常量改名（旧名字不再识别）：`singulator.station.WEIGH_MAX_S` → `sensing.weigher.WEIGH_MAX_S`，`singulator.station.STOP_BACK` → `machine.station.STOP_BACK`，`singulator.station.APPROACH` → `control.station.APPROACH`。`perception.SPLIT_PAD` / `GHOST_S` 设为 `None` 的旧行为已删除，只能在重构前的代码上跑。

**重构前的代码**在 git 标签 `before-refactor`（提交 `77c8907`），连同 S6 / S7 当时的源码快照、任务清单和过程记录；2026-10-06 起这些不在工作目录里，取用方法见 [README](../README.md#2026-10-05-结构重构)。

## 行为是否变了：怎么验证的

- **方法**：重构前（提交 `77c8907`）和重构后的代码，在同一台机器、同一套库版本上各跑同一批工况，逐位比较 `result.json`（去掉时间戳、输出目录、源码哈希）、轨迹（每 10 ms 全部记录量的哈希）和模型 XML。重构前的代码自己重复跑两遍，54 例全部逐位相同，所以比较中出现任何差异都是代码造成的。
- **结果：47 例全部逐位相同。** 覆盖：基线四种布料 14 例、`--sensing oracle` 1 例、相机控制给料 6 例、验收场景 6 例、故障注入 5 例、结构与控制选项 7 例（平板犁面、无侧带、车道规则、滚动摩擦、粗步长、`--set` 改常量等）、机头小试与滚筒机头 3 例、结构候选 5 例。这些工况里发生过犁面撤离（4 例，其中 2 例最终卡死停线）、给料点动、缓冲带停位、区段暂停、作废停住、件的拆分、全线冻结和各类故障停线。另有 29 组配置只比较模型 XML，全部相同。
- **补充（2026-10-06，Python 3.12，与项目 `.venv` 同一小版本）**：快速自检全部通过；用产生 S7 结论数据的那份源码快照和重构后的代码各跑同样 8 批（含 S7 记录里没测成的 5 批），`s7_compare.py` 比对：7 批逐项相同，另 1 批两边都因初始布料重叠被拒绝运行。那份快照和重构前的项目代码只差往结果里写 `measurement_audit` 一项。
- **比较时排除的字段**：`config` 里已删除的选项和新增的 `set`、`geometry.skew_deg`，以及两处提到旧模块名的说明文字（`provenance.limitations`、`geometry.devices.note`）。
- **有意的行为变化只有一处**：`--sensing oracle` 下的停滞判断并到了默认路径的同一套规则（`SectionWatch` 读理想视图）。试了一例 oracle 模式下发生停滞的批次，撤离时刻和停线时刻与重构前相同；这只是一例，不保证所有情况相同。默认路径（`--sensing vision`）不受影响。
- **没有验证的**：Windows 上的运行——验证在 Linux 上做，重构前同一种子在两个平台上的结果本来就有差异（种子 392 清线时刻 44.33 s 对 44.15 s），所以"等价"只在同一平台内成立；被删除的选项对应的旧结果，新代码无法复现，要用重构前的代码（git 标签 `before-refactor`）。视频渲染只跑通过一次（0.3 s，无中文字体）。
- 自检：`python checks/run_all.py --full`。在本机验收重构：用同样的种子重跑 S7（约 20 分钟），再用 `experiments/diff.py` 和保存的结果逐批比对，命令见 [experiments/README.md](../experiments/README.md)；预期逐批相同。

## 2026-10-06 提速：结果逐位不变

一次运行的时间原来大半花在 Python 上，不在 MuJoCo：种子 392 的前 20 s（8 万步）每步 544 µs，其中 `mj_step` 约 100 µs。改后约 233 µs。同机 4 进程跑下面那 14 批，总耗时 1224 → 558 s（单批快 1.9–2.5 倍，相贴布料和机头小试提得最少）。只改了"怎么算"，没改"算什么"：

- **数值筛查**（`physics/numerics.py`）：MuJoCo 的警告计数原来每步逐个读结构体，每步约 120 µs，比 `mj_step` 还慢；改读一个数组。每步的接触筛查用列表在 Python 里算，遇到落仓接触或非有限的距离才走原来的 numpy 路径。
- **驱动**（`physics/drives.py`、`actuators.py`、`control/station.py`）：每步对标量调用的 `np.clip`、`np.interp` 换成等值的 `min/max` 和缓存；侧带板条用切片写；各条带的载荷从一次 `tolist()` 里取。
- **视觉**（`sensing/vision.py`）：相机视图和给料控制器的理想视图共用同一帧算出的凸包和可见性；轨迹速度每个历史状态只拟合一次；单块料不再重复求凸包；跟踪匹配先在 Python 里排除明显不在轮廓里的轨迹（离边超过 pad + 1e-9），其余仍由 `inside()` 判；视场测试在矩阵乘法之后逐点比较。
- **记录与核验**（`sim/record.py`、`verify/station.py`、`sensing/weigher.py`、`physics/lumps.py`）：接触对一次取成列表，不再每个接触建一个结构体对象；块体包围盒每帧只算一次，光束和出口截面共用。

**怎么验证的**：改前（`4617fd9`）和改后的代码在同一台机器上跑同样 14 批——四种布料、`--feeder-stop beam`、相机给料（`--feeder-sensing vision --weigh-model steady`）、`--sensing oracle`、相贴和跨接缝验收场景、故障注入（`scan_fail`、`beam_dirty`）、机头小试、平板犁面、滚筒机头加扭转摩擦——逐位比较完整结果（去掉时间戳和源码哈希）、轨迹全部字段和模型 XML：14 批全部相同。改写过的几何函数（`clip`、`inside`、`in_view`、法向、面积、形心、速度拟合）另用随机输入和原函数逐位比较。快速自检和 `--full` 验收通过。和上一节一样，等价只在同一平台、同一套库版本内成立。

**以后改这些路径要注意**：每步（4000 次 / 仿真秒）和每帧（100 次 / 仿真秒）的代码里，一个 numpy 调用在 `mj_step` 之后要 2–10 µs，对十几个数的数组远比 Python 算术慢。求和、均值、点积、矩阵乘法、`lstsq` 要留在 numpy（换成 Python 会改变舍入，结果就不再逐位相同）；逐元素的加减乘除、比较、`min/max`、`sqrt` 用 Python 浮点算，结果相同。

**剩下的时间**：`mj_step` 约 43 %（碰撞约 34 µs，约束求解约 30 µs）；每帧的视觉约 1.7 ms（速度拟合、凸包、轮廓测量）；每步的驱动和筛查约 60 µs。再快要动数值设置（求解精度、步长、摩擦锥）或视觉的算法，结果会变，要另做验证。

## 还没解决的结构问题

- 几何字典 `d`（`machine/layout.py` 文档里列了它的键）仍是普通字典，没有类型约束。
- `control/station.py` 仍有约 750 行、一个类（重构前 1005 行）：件的归并 / 拆分、S2 交接、排料板路径、报警都在里面。它们共用同一份件列表，这次没有再拆。
- 犁面接触力（`FaceRetract.contact`）是真值量，仍记在控制器对象里，只进报告、不参与决策。
- `result.json` 的结构没有动，仍是一个大字典。
