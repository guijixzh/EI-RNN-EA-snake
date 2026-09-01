# test10_lunar 实验日志 —— LunarLander-v3 上的 EI-RNN（二度移植）

> 2026-08-27 ~ 08-29。基于 test7b 模板把 EI-RNN 进化移植到 gymnasium
> LunarLander-v3（8 维连续观测 / Discrete(4)），纠正 test5a_lunar 首次移植
> 的根本错误后交云端长跑。**本实验无结题结论**（云端长跑结果未归档回仓库）。

## 0. 继承关系与归档位置

```
test5a_lunar (首次移植, experiments/test5a_lunar.py, 败于 Bootstrap Threshold)
   └→ test10_lunar (test7b 模板重写, 本日志)
```

- **归档位置**：`experiments/test10_lunar/`（test10_lunar.py、断点
  test10_lunar_checkpoint.pth、云端部署包 `deploy/`，即原根目录
  deploy_test10_lunar/：README_AutoDL.md、run_test10_lunar.sh、
  requirements.txt 等）。
- 配套分析脚本：`experiments/measure_lunar_threshold.py`（通道阈值实测）、
  `experiments/diagnose_test10_noise.py`（器噪与盆地诊断）。

## 1. 主要目的

验证 test7b 已验证要素（GeneStack 整栈张量进化、E-I 双层动力学
BASE_TAU_E=0.7 / w_ei=w_ie=2.0 / N=256 / INIT_DENSITY=0.15、K=5 帧率思考、
FP16 分块全并行评估+显存自适应）能否**跨环境迁移**到连续控制任务。

## 2. 重要更新

1. **通道定标（纠正 test5a 的致命错误 "Bootstrap Threshold"）**：test5a 把
   8 维原始物理量直接进网，低幅通道（腿接触 bit、小坡度）振幅 << σ_win=0.1
   自举阈值，随机种群永远无法翻转 argmax、进化卡死。判据：输入信号振幅 ×
   权重初始化标准差 > 网络输出阈值。实测脚本测得归一化单位下各通道阈值
   a*≈1.6~3.2，据此定 `CHANNEL_SCALES = [6.4,3.2,6.4,6.4,3.2,6.4,3.2,3.2]`
   （results/test10_channel_threshold.json）。
2. **修复致命 bug（round-2 smoke 复查发现）**：`BatchedLunarEnv.step` 从未
   刷新 obs_np——网络每帧只看到 t=0 初观测，智能体全程失明，只能输出恒定
   动作序列（"集体自由落体"：58/60 坠毁、主引擎占空比 0.00、BestActs 恒
   [1/0/0/0]）。已改为跨个体逐步刷新观测。
3. **评估器噪声诊断**（diagnose_test10_noise.py）：2 局评估器噪声 std=30
   （P95-P5=97），与近 30 代 best-avg 差距 168 同量级；k=2 时精英排序
   Kendall-τ=0.24 / top-1 命中 0.18（≈随机），k=16 才可用。CRN 同种子
   256 槽散度=0：环境确定性成立，共公共随机数有效。
4. **适应度地形认知**：shaping 奖励快速接近发射台；点火/刹车立刻亏燃料 +
   shaping；转着降 +200 落在"刹车但仍旧坠毁"山谷另一侧——奖励地形多盆地。

## 3. 结果与结论

- **诊断时点（修复前）76 代状态**：avg=-143±5 平台、best 在 [-17,132]
  震荡（30 代均值 25.1、std 33.7）；单一策略普查 60 局：及格着陆 0 /
  软着陆 2 / 坠毁 58（results/test10_noise_diagnosis.{json,png}）。
- **结论**：失明 bug、评估器噪声与多盆地地形三者叠加导致平台；第 2 轮
  改造（修 bug + k=16 精评 + CRN）完成后于 2026-08-29 打包交 AutoDL
  云端长跑（`experiments/test10_lunar/deploy/`）。
- **状态**：云端训练结果未归档回本仓库（results/ 无曲线产物），实验
  无结题结论；如需收尾，先取回云端 history 再续写本日志。
