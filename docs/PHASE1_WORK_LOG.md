# Phase 1 提取工作日志

## 日期：2025年（本次会话）

---

## ✅ 今天完成的工作

### 1. 发现并诊断质量问题

**问题发现：**
- 运行Phase 1提取（170个电影，5%采样）
- 创建分析notebook：`notebooks/analyze_phase1_results.ipynb`
- 发现两大严重问题：
  1. **Prompt依赖100%**：所有relation都来自prompt的8个例子
  2. **Entity多样性80%**：几乎每个电影的entity都不同，图谱极度稀疏

**质量诊断结果：**
```
Relation多样性: 8个（警告：类型过少）
Prompt依赖度: 100.0%（警告：严重依赖）
Entity多样性: 997个唯一entity，平均复用2.14次（注意：偏高）
平均知识点: 12.6（良好）
```

---

### 2. 重新设计Prompt

**文件修改：** `src/extraction/prompts.py`

**主要改进：**

#### A. 解决Relation依赖问题（100% → 17.7%）
- ❌ 删除：具体的8个relation列表（color_scheme, visual_style等）
- ❌ 删除：固定格式的例子
- ✅ 添加：原则性指导，鼓励发现多样relation
- ✅ 添加：抽象的格式例子（不限制类型）

#### B. 解决Entity稀疏问题（80% → 65%）
- ✅ 强调：使用ABSTRACT、HIGH-LEVEL术语
- ✅ 提供：好/坏entity对比例子
- ✅ 引导：思考"类别"而非"具体描述"
- ✅ 要求：entity应该适用于多个电影

**关键Prompt改进：**
```
**CRITICAL Guidelines for Entities:**
1. Use ABSTRACT, HIGH-LEVEL terms that can apply to MULTIPLE movies
2. Avoid overly specific descriptions - think in CATEGORIES, not unique details
3. Prefer COMMON visual terms over rare combinations
4. Ask yourself: "Could this entity describe other movies too?"

**Good Entity Examples (abstract, reusable):**
- warm_tones, cool_tones, monochrome
- human_portrait, action_scene, landscape
- minimalist, vintage, dramatic

**Bad Entity Examples (too specific, hard to reuse):**
- sunset_over_ocean_with_sailboat
- woman_in_red_dress_holding_gun
```

---

### 3. 创建测试脚本并验证改进

**新增文件：** `scripts/test_new_prompt.py`

**测试结果（10个随机电影）：**

| 指标 | 旧Prompt | 新Prompt | 改进 |
|------|---------|---------|------|
| **Relation多样性** | 8 | **43** | **+35** ✅ |
| **Prompt依赖度** | 100% | **17.7%** | **-82.3%** ✅ |
| **Entity复用率** | 1.28x | **1.53x** | +0.25x ⚠️ |

**新发现的Relation示例：**
- dominant_color, artistic_style, depicted_subject
- character_type, visual_theme, text_style
- emotional_tone, background_type, layout
- symbolic_element, vehicle_type, geographical_setting
- 等43种（vs 旧版8种）

**结论：**
- ✅ Relation问题完全解决
- ⚠️ Entity有改善但仍偏稀疏（可接受，Phase 2聚类会解决）

---

### 4. 修复增量提取逻辑

**文件修改：** `scripts/run_phase1_extraction.py`

**问题修复：**
- 默认跳过**所有**已处理ID（success + error），避免重复累积
- 添加 `--retry-errors` 参数支持重试失败项
- 移除旧error记录再重新处理，避免重复

**实时保存机制：**
- 每处理完一个电影立即保存结果
- 使用原子写入（temp file + rename）防止损坏
- 支持随时中断和恢复

---

## 📊 提交记录

```
2d6d783 Redesign Phase 1 prompt to fix relation & entity quality issues
0072d37 Add Phase 1 quality analysis notebook
140dbb0 Fix incremental extraction logic: properly skip all processed IDs
d0a70aa Add real-time incremental saving to Phase 1 extraction
ebdbf32 Revert: require explicit api_key for custom endpoints
```

---

## 🎯 下一步计划

### 立即执行：
1. **重新提取170个电影**（使用新prompt）
   ```bash
   python scripts/run_phase1_extraction.py \
       --poster_dir data/raw/ml-1m/posters \
       --id_mapping data/recbole/ml-1m/id_mappings.json \
       --base_url http://10.12.208.86:8502 \
       --api_key "YOUR_KEY" \
       --percentage 5.0 \
       --seed 42
   ```

2. **分析新结果**
   - 运行 `notebooks/analyze_phase1_results.ipynb`
   - 验证Relation多样性（期望：40-60种）
   - 验证Prompt依赖度（期望：<30%）
   - 验证Entity分布

### 后续阶段：

3. **Phase 2: 聚类与词汇表构建**
   - Relation聚类：选出10-15个最有用的
   - Entity聚类：每个relation下保留Top 20-30个
   - 构建受控词汇表（含"others"）

4. **Phase 3: 20%验证提取**
   - 使用受控词汇表
   - 约680个电影（3416的20%）
   - 验证词汇表覆盖率

5. **Phase 4: 80%全量提取**
   - 约2733个电影
   - 构建完整知识图谱

---

## 💡 关键设计决策

### 为什么接受Entity稍高的稀疏性？

**理由：**
1. Phase 1是**探索阶段**，目标是发现所有可能的entity
2. Phase 2会通过**聚类**筛选高频entity
3. Phase 3/4会使用**受控词汇表**，长尾entity映射到"others"
4. 最终图谱的entity复用率会很高（预期4-8x）

**流程示例：**
```
Phase 1: glamorous_femme_fatale (2次), intimate_connection (3次),
         human_portrait (80次)
         → 允许长尾entity存在

Phase 2: 聚类筛选 → 只有human_portrait进入词汇表
         → 低频entity被过滤

Phase 3: 强制词汇表 → femme_fatale映射到character_type: others
                    → human_portrait映射到character_type: human_portrait
         → 高复用率✅
```

---

## 📁 关键文件

### 修改的文件：
- `src/extraction/prompts.py` - 重新设计的prompt
- `scripts/run_phase1_extraction.py` - 增量提取与实时保存

### 新增的文件：
- `notebooks/analyze_phase1_results.ipynb` - 质量分析notebook
- `scripts/test_new_prompt.py` - Prompt测试脚本

### 结果文件：
- `results/phase1_5percent_exploration.json` - 待更新（新prompt）
- `results/phase1_5percent_exploration_old_prompt.json` - 旧版本备份
- `results/prompt_test_new.json` - 测试结果（10个电影）

---

## 📈 预期改进（基于测试）

重新提取170个电影后，预期：

| 指标 | 旧版 | 新版（预期） |
|------|------|-------------|
| 唯一Relation数 | 8 | **50-70** |
| Prompt依赖度 | 100% | **<25%** |
| 唯一Entity数 | 997 | **1000-1200** |
| Entity平均复用 | 2.14x | **1.8-2.0x** |
| Top 10 Entity复用 | - | **30-80次** |

**对推荐系统的影响：**
- ✅ 更丰富的relation维度（50+ vs 8）
- ✅ Top entity形成强连接（如human_portrait可能连接100+电影）
- ✅ 保留探索性，Phase 2聚类会进一步优化

---

## 🔧 技术改进

1. **实时保存机制** - 每个电影处理完立即保存
2. **原子写入** - 防止文件损坏
3. **增量提取** - 支持中断恢复
4. **错误重试** - `--retry-errors` 参数
5. **质量分析** - 自动化notebook分析

---

## ✅ 验收标准

新提取完成后，需验证：
- [ ] Relation数量：40-70个 ✅
- [ ] Prompt依赖度：<30% ✅
- [ ] 平均KP数量：10-15个 ✅
- [ ] Top 20 entity复用率：>10次 ✅
- [ ] 所有170个电影成功提取 ✅

达标后进入Phase 2聚类阶段。
