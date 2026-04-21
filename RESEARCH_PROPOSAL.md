# 面向多模态驾驶世界模型的研究计划

## 0. 一句话版本

本项目不再把重点放在手工 reward shaping 或简单场景堆叠上，而是研究：

**如何让 Dreamer 类世界模型在多模态驾驶场景中学到可分离、可生成、可干预的 latent 表示，并通过 Conditional Flow Matching 与反事实课程学习抑制多模态规划中的危险模态选择。**

拟定方法名：

**CFM-CounterDreamer: Counterfactual Curriculum Flow Matching for Multimodal Latent Driving World Models**

## 1. 问题动机

自动驾驶中的多模态不是一个附加现象，而是闭环决策的核心困难。相同观测下，未来可能存在多个合理行为：

1. 前车慢行时，可以跟车、换道或提前减速。
2. 匝道汇入时，可以抢先汇入、等待空隙或让行。
3. 路口交互中，其他车辆意图不确定， ego 的安全动作也不唯一。

传统 Dreamer/RSSM 往往把这种不确定性压进一个单一 latent 动力学里，训练目标又偏向重建和 reward 预测，容易出现两个问题：

1. **模态平均化**：latent 表示混合多个未来，生成的轨迹看似平滑但语义上不合理。
2. **差模态放大**：多模态策略中某些低概率但高风险的模态会在闭环中被 actor 或价值函数误选，导致碰撞、抢行、犹豫或越界。

因此，本研究的核心不是“生成更多轨迹”，而是让模型学会：

1. 哪些 latent 模态对应可行驾驶意图。
2. 哪些模态虽然能解释数据，但在闭环中危险。
3. 如何通过反事实课程把危险模态暴露出来并校正策略。

## 2. 核心假设

本研究基于三个假设：

1. **CFM 比单纯 KL/RSSM 更适合学习多模态 latent 转移。**
   Conditional Flow Matching 可以学习从简单噪声分布到未来 latent 分布的连续传输路径，比单高斯先验或离散 latent 更自然地表达多峰未来。

2. **多模态驾驶的失败往往来自 latent 模态选择错误，而不是单步控制误差。**
   例如模型知道“可以让行”也知道“可以抢行”，但策略在临界状态下选择了危险模态。

3. **反事实课程可以专门攻击这些错误模态。**
   通过对速度、间距、cut-in 时间、他车意图进行反事实扰动，可以把模态边界附近的坏行为系统性放大，使模型学会拒绝或降权危险 latent。

## 3. 方法总览

整体框架分为五个模块：

1. **多模态 latent 世界模型**
   在 Dreamer/RSSM 基础上保留 recurrent state，但引入显式 latent mode 表示，用于描述未来意图和交互模式。

2. **Anchor-conditioned Conditional Flow Matching latent transition**
   不把锚点作为最终轨迹模板，而是把锚点作为粗模态条件，再用 CFM 学习连续 latent residual 分布：

   ```text
   z_t, a_t, context_t, anchor_k, noise -> z_{t+1:t+H}
   ```

   其中 `context_t` 包含 BEV/occupancy、导航目标、邻车状态、历史轨迹等多模态输入。

3. **模态风险评估与安全选择**
   对每个生成 latent 模态预测任务收益、碰撞/越界/TTC 风险、舒适性和不确定性，然后选择满足安全约束的模态执行。

4. **动态曲率可行性约束**
   对每条解码轨迹计算动态曲率上界，过滤高速急转、不可执行变道、过大横向加速度等动力学不可行模态。

5. **反事实课程学习**
   从失败或高不确定片段中构造反事实样本，逐步提高难度，让模型在训练中反复遇到“容易选错模态”的状态。

## 4. 方法细节

### 4.1 多模态 latent 表示

令世界模型的状态为：

```text
s_t = (h_t, z_t, m_t)
```

其中：

1. `h_t` 是 RSSM deterministic recurrent state。
2. `z_t` 是连续 stochastic latent，编码局部动力学和场景不确定性。
3. `m_t` 是 mode embedding，编码驾驶意图，例如跟车、让行、换道、汇入、避让。

`m_t` 不需要人工标注，可以通过以下弱监督得到：

1. 轨迹几何聚类：横向位移、速度变化、相对车距变化。
2. 规则伪标签：lane keep、lane change、yield、overtake、merge。
3. CFM latent 聚类：根据未来 latent rollout 的拓扑结构自动分簇。

目标不是得到完美语义标签，而是让世界模型内部有一个可操作的多模态结构。

### 4.2 锚点条件化的 latent CFM

现有多模态轨迹研究常用 anchor 生成多条候选轨迹，例如 endpoint anchor、maneuver anchor 或聚类得到的轨迹原型。它的优势是稳定、可解释、不容易 mode collapse；缺点是容易退化成“模板选择 + residual 回归”，长尾交互和连续模态边界表达能力有限。

本方案不采用 hard anchor planner，而采用：

```text
anchor-conditioned latent CFM
```

也就是说，anchor 只提供粗模态条件，不直接决定最终轨迹：

```text
context_t + anchor_k + noise -> CFM -> future latent z_{t+1:t+H}
```

或者等价地：

```text
anchor_k 给出粗意图
CFM 生成该意图下的连续 latent residual
```

anchor 可以有两种来源：

1. **trajectory anchor**
   在 NAVSIM/nuPlan/Waymo 的未来轨迹上聚类，得到 lane keep、slow down、left shift、right shift、merge、yield、overtake 等粗模态。

2. **latent anchor**
   先用 Dreamer posterior 得到 `z_{t:t+H}`，再对 latent sequence 聚类，得到更贴近世界模型内部动力学的 mode token。

主方法优先采用 latent anchor，因为它与 latent CFM 的叙事一致；trajectory anchor 作为强 baseline 和可解释性参照。

### 4.3 Conditional Flow Matching for Latent Dynamics

标准 Dreamer 的 prior 通常学习：

```text
p(z_t | h_t)
```

本方案改为学习未来 latent 分布：

```text
p(z_{t+1:t+H} | h_t, z_t, a_t, o_t, g_t, anchor_k)
```

通过 CFM 训练一个 velocity field：

```text
v_theta(x_tau, tau | condition)
```

其中 `x_tau` 是从噪声 latent 到真实未来 latent 的插值点，`tau` 是 flow time，`condition` 是当前世界模型状态、动作、导航和场景上下文。

训练损失：

```text
L_cfm = E[ || v_theta(x_tau, tau | c_t, anchor_k) - u_tau ||^2 ]
```

直观上，模型学习“如何从随机 latent 运输到一个合理未来”。推理时采样多个噪声，就能得到多个未来 latent 模态。

训练和推理可以支持三种模式：

1. **Anchor-free CFM**
   `context + noise -> latent future`，用于检验 CFM 本身的多模态能力。

2. **Hard anchor baseline**
   `context -> anchor probability + trajectory residual`，用于对比传统 anchor 方法。

3. **Ours: Anchor-conditioned latent CFM**
   `context + anchor token + noise -> latent future residual`，作为主方法。

### 4.4 多模态规划与风险选择

对每个候选 latent 模态 `k`，解码出：

1. 未来 ego trajectory。
2. occupancy 或周围 agent rollout。
3. reward/value estimate。
4. risk estimate：collision、offroad、lane violation、TTC、jerk。
5. epistemic uncertainty：例如 ensemble variance 或 flow sample disagreement。
6. feasibility violation：动态曲率、横向加速度、转向变化率等可行性违约。

执行时不直接选 value 最大的模态，而采用：

```text
argmax_k V_k
s.t. Risk_k <= budget, Uncertainty_k <= threshold, Feasibility_k <= threshold
```

如果所有模态都不安全，则进入保守 fallback，例如减速、让行或 expert intervention。

这部分的论文卖点是：多模态生成不只是为了覆盖未来，而是为了在闭环中进行风险感知的模态选择。

### 4.5 动态曲率可行性上界

为了避免 CFM 生成“看起来能到目标、但动力学不可执行”的坏模态，引入基于轨迹拟合曲率的动态适应上界。

对每条候选轨迹拟合或离散估计曲率：

```text
kappa(t) = trajectory_curvature(x_t, y_t)
```

并根据车辆状态和道路上下文计算动态上界：

```text
kappa_max(t) = min(vehicle_limit, comfort_limit, road_adaptive_limit)
```

其中可用近似形式为：

```text
vehicle_limit = tan(delta_max) / L
comfort_limit = a_lat_max / max(v_t^2, eps)
road_adaptive_limit = |kappa_road(t)| + margin(v_t, road_type)
```

曲率违约定义为：

```text
L_curv = mean ReLU(|kappa(t)| - kappa_max(t))^2
```

它在系统中承担三类作用：

1. **训练约束**
   作为 CFM decoder 或 trajectory decoder 的 regularization，抑制高速急转、不可执行变道和过大横向加速度。

2. **风险标签**
   将 `curvature_violation`、`lateral_acc_violation`、`steering_rate_violation` 加入 risk head 的监督信号。

3. **模态过滤**
   推理时先过滤动力学不可行模态，再在剩余候选中做 value/risk 选择。

这使 bad mode 不只包含碰撞/越界，也包含“物理上不合理或舒适性不可接受”的生成模态。

### 4.6 反事实课程学习

反事实课程的目标是专门制造“多模态会犯错”的训练样本。

从 replay buffer 中筛选以下片段：

1. 碰撞前 1-3 秒。
2. TTC 低但未碰撞的 near miss。
3. actor/value 对多个模态评分接近的状态。
4. flow samples 分歧大的状态。
5. expert intervention 或 risk trigger 发生前后的状态。
6. 曲率或横向加速度接近上界的状态。

对这些状态构造反事实扰动：

1. 改变他车速度：更快逼近、更慢阻塞。
2. 改变相对位置：缩小 gap、制造盲区。
3. 改变 cut-in 或 merge 时机。
4. 改变交通密度和 aggressiveness。
5. 改变导航目标或可行车道结构。
6. 改变道路曲率、车道宽度或可用 gap，使原本可行的 anchor 变成危险/不可行模态。

课程难度由一个评分器控制：

```text
difficulty = alpha * risk + beta * model_uncertainty + gamma * mode_ambiguity
           + eta * feasibility_violation
```

训练从低难度反事实开始，逐步进入高风险、高歧义、长尾交互样本。

### 4.7 对抗多模态差行为

“差行为”定义为：模型生成或策略选择了数据中可能出现、但在当前上下文下闭环危险的模态。

例如：

1. 在 gap 不足时选择汇入。
2. 在前车急刹时选择保持高速跟车。
3. 在邻车 cut-in 时选择继续加速。
4. 在多车交互中反复犹豫，导致停滞或被追尾。
5. 高速下生成大曲率急转或不可执行避障轨迹。

抑制方式：

1. **bad-mode contrastive loss**
   对安全模态和危险模态做对比学习，使 latent mode 在表示空间中分离。

2. **counterfactual risk ranking**
   同一原始场景与反事实场景之间，要求模型给更危险的反事实模态更高 risk。

3. **mode dropout**
   训练时随机隐藏最优模态，迫使模型学习多个可行备选，而不是只记住单一 expert 行为。

4. **closed-loop rejection loss**
   对导致闭环失败的 sampled mode，降低其被 actor 选择的概率。

5. **feasibility rejection loss**
   对超过动态曲率上界或横向加速度上界的模态，降低其采样概率或选择概率。

## 5. 与当前工程的关系

当前项目已经有 DreamerV3/MetaDrive/NAVSIM 相关工程基础。建议把新研究分成三个落地点。

### 5.1 世界模型侧

重点位置：

1. `dreamer/embodied/agents/`
2. Dreamer world model 中 RSSM prior/posterior 的实现。
3. replay buffer 中轨迹片段采样逻辑。

新增能力：

1. latent sequence sampler。
2. CFM velocity network。
3. anchor/mode embedding head。
4. trajectory/occupancy/risk decoder。
5. curvature feasibility checker。

### 5.2 环境和数据侧

重点位置：

1. `dreamer/embodied/envs/metadrive_lane_keeping.py`
2. `dreamer/embodied/envs/metadrive_on_ramp.py`
3. `dreamer/embodied/envs/metadrive_lane_reduction.py`
4. NAVSIM mini 数据管线。

新增输出：

1. agent states：位置、速度、航向、相对距离。
2. risk event：collision、offroad、lane violation、TTC。
3. intervention flag：expert takeover 或 risk_trigger。
4. scenario context：road type、traffic density、merge/cut-in 标签。
5. road geometry：道路中心线、道路曲率、车道宽度和可行驶边界。

### 5.3 训练侧

建议训练流程：

1. **Stage A: IL/DAgger warmup**
   先得到可运行策略和初始 replay。

2. **Stage B: CFM latent world model**
   用 replay 训练多模态 latent transition。

3. **Stage C: risk-aware actor training**
   在 sampled latent rollout 中训练 actor，但加入模态风险选择。

4. **Stage D: counterfactual curriculum**
   持续从失败片段生成反事实课程，重训或微调 world model 与 actor。

5. **Stage E: feasibility-aware mode rejection**
   在训练和推理中加入动态曲率上界，过滤不可执行模态并记录 bad-mode selection rate。

## 6. 论文创新点

### 6.1 Anchor-conditioned latent CFM world model

不是把 flow matching 用作轨迹后处理，也不是直接用 hard anchor 输出轨迹，而是把 anchor 作为粗模态条件放入 Dreamer latent dynamics 内部，用 CFM 学习连续多峰 latent residual 分布。

### 6.2 Bad-mode-aware multimodal planning

显式区分“合理模态”“危险模态”和“不确定模态”，避免多模态模型只追求覆盖率而忽视闭环选择错误。

### 6.3 Dynamic feasibility-aware mode rejection

引入基于速度、车辆约束、道路曲率和舒适性的动态曲率上界，将动力学不可行的生成模态纳入 bad mode，并在训练和推理中显式过滤。

### 6.4 Counterfactual curriculum for mode robustness

反事实课程不是普通数据增强，而是围绕模态歧义、risk trigger 和闭环失败生成训练样本，专门提升多模态边界处的鲁棒性。

### 6.5 Unified evaluation of multimodal latent quality and closed-loop safety

同时评估 latent 表示、未来分布覆盖、风险排序、闭环安全，而不是只报告 reward 或 success rate。

## 7. 实验设计

### 7.1 数据和环境

第一阶段使用当前最容易跑通的环境：

1. MetaDrive lane keeping。
2. MetaDrive lane reduction。
3. MetaDrive on-ramp merge。

第二阶段迁移到更有论文说服力的数据：

1. NAVSIM mini/full。
2. nuPlan 或 Waymo Motion 作为离线多模态预训练。
3. Bench2Drive 作为闭环泛化评测。

### 7.2 Baselines

1. DreamerV3 baseline。
2. Dreamer + DAgger/IL。
3. Hard anchor trajectory planner。
4. Dreamer + trajectory diffusion planner。
5. Dreamer + CFM trajectory planner，但不做 latent CFM。
6. Anchor-free latent CFM。
7. Ours without counterfactual curriculum。
8. Ours without dynamic curvature filtering。
9. Ours without bad-mode rejection。
10. Ours full model。

### 7.3 Metrics

闭环指标：

1. success rate。
2. route completion。
3. collision rate。
4. offroad rate。
5. lane violation。
6. comfort：jerk、steering smoothness。
7. intervention count。

多模态指标：

1. minADE/minFDE。
2. mode coverage。
3. negative log likelihood 或 flow matching validation loss。
4. mode collapse rate。
5. bad-mode selection rate。
6. risk ranking accuracy。
7. safe-mode coverage。
8. anchor utilization entropy。

反事实鲁棒性指标：

1. counterfactual success rate。
2. near-miss recovery rate。
3. TTC violation under perturbation。
4. performance drop from original to counterfactual scenes。
5. curvature violation under perturbation。

### 7.4 关键消融

1. RSSM prior vs CFM latent prior。
2. trajectory-space CFM vs latent-space CFM。
3. hard anchor planner vs anchor-conditioned latent CFM。
4. trajectory anchor vs latent anchor。
5. no mode embedding vs learned mode embedding。
6. no dynamic curvature bound vs static curvature bound vs adaptive curvature bound。
7. no counterfactual curriculum vs random augmentation vs targeted counterfactual curriculum。
8. no bad-mode contrastive loss。
9. no risk-aware mode selection。
10. different number of flow samples K。

## 8. 预期图表

论文中至少需要以下图表：

1. 方法总览图：Dreamer latent state、CFM sampler、risk selector、counterfactual curriculum。
2. latent t-SNE/UMAP：不同 driving modes 是否分离。
3. 多模态轨迹可视化：正常场景与反事实场景对比。
4. risk ranking 曲线：危险模态是否被正确排序。
5. bad-mode selection rate 随训练下降曲线。
6. 原始场景和反事实场景的闭环指标柱状图。
7. flow samples 数量 K 与性能/延迟折中曲线。
8. 曲率上界可视化：不同速度/道路曲率下的可行轨迹过滤效果。
9. anchor 使用分布：hard anchor、anchor-free CFM、anchor-conditioned latent CFM 的模态覆盖对比。

## 9. 8 周执行计划

### 周 1：整理数据接口和失败片段

1. 统一 MetaDrive/NAVSIM observation 到多模态 context。
2. 记录 risk event、TTC、near miss、expert intervention。
3. 从 replay 中导出失败片段和临界状态。

### 周 2：实现 latent trajectory dataset

1. 从 Dreamer replay 中采样 `z_{t:t+H}`。
2. 增加 mode pseudo-label。
3. 构造 trajectory anchor 和 latent anchor。
4. 构造 CFM 训练 batch。

### 周 3：实现 CFM latent transition

1. 实现 velocity network。
2. 训练 anchor-free latent CFM。
3. 训练 anchor-conditioned latent CFM。
4. 对比 RSSM prior、hard anchor 和 CFM prior 的 rollout quality。

### 周 4：多模态 decoder 与风险头

1. 解码 K 个 latent samples 到 trajectory/occupancy。
2. 训练 risk head。
3. 实现动态曲率上界和 feasibility checker。
4. 实现 risk-aware + feasibility-aware mode selection。

### 周 5：反事实样本生成

1. 基于失败片段生成速度、位置、gap、cut-in 时机扰动。
2. 实现 difficulty scorer。
3. 加入 curriculum scheduler。

### 周 6：bad-mode 对抗训练

1. 实现 bad-mode contrastive loss。
2. 实现 counterfactual risk ranking loss。
3. 实现 feasibility rejection loss。
4. 统计 bad-mode selection rate 和 curvature violation rate。

### 周 7：主实验和消融

1. 完整模型跑 MetaDrive 三类场景。
2. 跑关键 ablation。
3. 输出所有闭环和多模态指标。

### 周 8：NAVSIM/Bench2Drive 迁移与论文材料

1. 在 NAVSIM mini 上做离线预训练或验证。
2. 补可视化图。
3. 写方法、实验、消融和失败分析。

## 10. Go/No-Go 标准

继续推进投稿版本需要满足：

1. CFM latent prior 的 minFDE 或 rollout consistency 明显优于 RSSM prior。
2. bad-mode selection rate 相比无反事实课程下降至少 20%。
3. 反事实场景下 collision/offroad 明显下降。
4. 原始场景性能不能因为保守选择显著退化。
5. 动态曲率过滤显著降低 curvature violation，且不明显降低 safe-mode coverage。
6. 推理延迟在可接受范围内，例如 K=8 或 K=16 samples 时仍能闭环运行。

如果 CFM latent 训练不稳定，备选方案是：

1. 先做 trajectory-space CFM 作为弱版本。
2. 使用 diffusion/flow 只生成 future latent residual。
3. 保留 RSSM prior，但在 actor 前加入 CFM mode proposal。
4. 若 latent anchor 不稳定，则先使用 trajectory anchor 作为 mode token。

## 11. 最小可行版本

最小可行版本不要一上来追求完整自动驾驶 benchmark。建议先完成：

1. MetaDrive on-ramp merge。
2. Dreamer replay 中提取 latent sequence。
3. 从 NAVSIM 或 replay 中聚类得到 trajectory/latent anchors。
4. Anchor-conditioned CFM 生成 K 个 future latent。
5. risk head 识别危险汇入模态。
6. 动态曲率上界过滤不可执行急转/抢行模态。
7. 反事实扰动 gap 和他车速度。
8. 证明 bad-mode selection rate、curvature violation rate 和 collision rate 下降。

这个 MVP 已经能支撑一个清晰故事：

**多模态驾驶失败来自错误 latent 模态选择；anchor 提供粗意图，CFM 学到连续多峰 latent 未来，动态曲率约束过滤不可行模态，反事实课程暴露危险模态，风险选择机制让策略在闭环中避开坏模态。**

## 12. 当前 TODO

- [ ] 梳理 Dreamer world model 中可插入 CFM latent prior 的代码位置。
- [ ] 增加 replay latent sequence 导出脚本。
- [ ] 从 NAVSIM/replay 中构造 trajectory anchor 和 latent anchor。
- [ ] 在 MetaDrive on-ramp 上构造第一版反事实 gap/速度扰动。
- [ ] 实现 anchor-conditioned CFM velocity network 与 latent sampler。
- [ ] 实现 mode pseudo-label 和 bad-mode 标注逻辑。
- [ ] 实现动态曲率上界、曲率违约指标和 feasibility checker。
- [ ] 实现 risk head 与 risk-aware mode selection。
- [ ] 实现 feasibility-aware mode rejection。
- [ ] 跑 RSSM prior vs CFM latent prior 的第一轮对比。
- [ ] 跑 hard anchor vs anchor-free CFM vs anchor-conditioned latent CFM 的消融。
- [ ] 输出 bad-mode selection rate、curvature violation rate、collision rate、counterfactual robustness 四组核心结果。
