# Phase 5: User Interest Extraction

*Created: 2026-01-12*
*Status: ✅ 完成*

---

## 📋 概述

Phase 5实现基于用户评分历史的兴趣提取，生成用户知识图谱（User KG），与电影知识图谱（Item KG）合并，用于知识增强推荐模型。

**核心目标**：
- 从用户评分历史中提取短期兴趣（Short-term Interests）和长期兴趣（Long-term Interests）
- 使用LLM作为information extractor（支持论文narrative）
- 生成RecBole格式的用户KG和合并KG
- 支持消融实验（Item KG, User KG, Merged KG）

---

## 🎯 设计策略

### **Hybrid Statistical + LLM Approach**

```
评分历史
   ↓
按活跃天数分桶（21天/桶）
   ↓
每个桶：统计聚合 → 短期兴趣（Top-10）
   ↓
每4个桶（84天）：LLM总结 → 长期兴趣（Top-5 with age）
   ↓
最终输出：5个长期 + 10个短期
```

**关键参数**：
- `SHORT_TERM_DAYS = 21`: 短期兴趣桶大小
- `LONG_TERM_BUCKETS = 4`: 每4个短期桶进行一次LLM总结
- `MIN_RATING = 4.0`: 只统计高分电影（≥4星）
- `SHORT_TERM_TOP_K = 10`: 每个短期桶保留Top-10（出现次数>1）
- `LONG_TERM_TOP_K = 5`: 长期兴趣保留Top-5

**理论依据**：
- 21天：习惯养成周期
- 84天（3个月）：季度总结周期
- 长期兴趣带年龄标记：追踪兴趣持续时间

---

## 💡 技术决策

### **决策1：为什么用LLM？**

**要求**：论文narrative需要说"LLM is a good information extractor"

**方案**：Hybrid approach
- **短期兴趣**：统计聚合（快速、免费、准确）
- **长期兴趣**：LLM推理（识别持续模式、更新年龄）

**优势**：
- 成本可控（只在高层总结时用LLM）
- LLM用于推理任务（而非简单统计）
- 支持消融实验（统计 vs LLM）

### **决策2：时序分桶策略**

**挑战**：ML-1M数据集的时间戳问题
- 59%用户在注册日批量评分（timestamps = 提交时间，非观影时间）
- 中位数活跃天数 = 1天

**决策**：按活跃天数分桶（而非日历时间）

**理由**：
1. 方法论正确（适用于更好的数据集如Amazon）
2. ML-1M退化为"整体兴趣提取"（大多数用户只有1次LLM调用）
3. 保持策略一致性，数据集问题不影响方法设计

### **决策3：词表约束**

**策略**：LLM只能从短期兴趣中选择entities

**实现**：
- Prompt中列出最近短期兴趣
- 明确规则："ONLY select from relation-entity pairs shown in short-term interests"
- 避免LLM幻觉

**优势**：
- 保证长期兴趣来自实际评分历史
- 与电影KG的entity词表自然对齐
- 可追溯性强

---

## 🏗️ 实现架构

### **代码组织**

```
src/extraction/user_interest_extractor.py  ← 核心逻辑（可复用模块）
    ├─ UserInterestExtractor (class)
    │   ├─ extract_short_term_interest()      # 统计聚合
    │   ├─ get_llm_summarization_prompt()     # Prompt生成
    │   ├─ summarize_long_term_interests()    # LLM调用
    │   └─ extract_user_interests()           # 完整流程
    ├─ load_movie_kg()
    └─ load_user_ratings()

scripts/extract_user_interests_hybrid.py   ← CLI包装器
    ├─ 参数解析
    ├─ 多线程调度
    └─ 结果保存

scripts/convert_user_interests_to_kg.py    ← RecBole格式转换
    ├─ convert_user_interests_to_kg()
    ├─ merge_item_and_user_kg()
    └─ CLI入口
```

### **数据流**

```
输入：
  - ml-1m.inter (评分数据)
  - phase4_full_extraction_filtered.json (电影KG)

处理：
  1. 加载电影KG (item_id → [(relation, entity), ...])
  2. 加载用户评分 (user_id → [interactions sorted by time])
  3. 按活跃天数分桶
  4. 每桶统计短期兴趣
  5. 每4桶LLM总结长期兴趣
  6. 输出JSON

输出：
  - user_interests_hybrid.json (原始提取结果)
  - ml-1m.user.kg (RecBole用户KG)
  - ml-1m.kg (合并KG)
```

---

## 📝 LLM Prompt设计

### **Prompt结构**

```
用户最近短期兴趣（过去4个时间段，84天）
   ├─ Period 1 (21天): Top-5 interests with counts
   ├─ Period 2 (21天): ...
   ├─ Period 3 (21天): ...
   └─ Period 4 (21天): ...

用户当前长期兴趣（if any）
   └─ 每个兴趣的 (relation, entity, age_days)

任务：更新长期兴趣
   ├─ 保留稳定的兴趣（age增加84天）
   ├─ 添加新出现的模式（age = 84天）
   └─ 移除不再相关的兴趣
```

### **关键Prompt规则**

#### **规则1：精确字符串匹配**
```
Use EXACT relation and entity strings from short-term interests above
- DO NOT modify relation names (composition_styles → composition_styles ✅)
- DO NOT simplify entities (bright lighting → bright lighting ✅)
- Copy-paste exactly as shown
```

**原因**：避免LLM改写导致词表不匹配

#### **规则2：年龄计算**
```
For interests from CURRENT LONG-TERM that you want to KEEP:
  age_days = their_current_age + 84

For interests from SHORT-TERM that are NEW (not in current long-term):
  age_days = 84

Example:
  - Current: "mood: romantic" (age=84) → Keep → new age = 168
  - Short-term: "genre: drama" (NEW) → Add → age = 84
```

**原因**：追踪兴趣持续时间（用于后续推荐模型）

#### **规则3：词表约束**
```
Only select from relation-entity pairs shown in short-term interests
```

**原因**：防止LLM幻觉，保证可追溯性

### **输出格式**

```json
[
  {"relation": "<exact_relation>", "entity": "<exact_entity>", "age_days": <number>},
  {"relation": "<exact_relation>", "entity": "<exact_entity>", "age_days": <number>},
  ...
]
```

**验证**：
- Exactly 5 items
- All relations/entities exist in short-term history
- Age calculated correctly

---

## 🚀 使用指南

### **Step 1: 提取用户兴趣**

```bash
python scripts/extract_user_interests_hybrid.py \
  --ratings data/recbole/ml-1m/ml-1m.inter \
  --movie_kg results/phase4_full_extraction_filtered.json \
  --output results/user_interests_hybrid.json \
  --model gpt-4o-mini \
  --workers 15
```

**参数说明**：
- `--ratings`: RecBole格式的交互数据
- `--movie_kg`: Phase 4电影KG提取结果
- `--model`: OpenAI模型（推荐gpt-4o-mini）
- `--workers`: 并发线程数
- `--test_users N`: 测试模式（只处理N个用户）

**预期运行时间**：
- 6,040用户：约10-15分钟（15 workers）
- 6,056次LLM调用

**预期成本**：
- gpt-4o-mini: $6-12
- 平均1.0次LLM调用/用户

---

### **Step 2: 转换为RecBole KG格式**

```bash
python scripts/convert_user_interests_to_kg.py \
  --input results/user_interests_hybrid.json \
  --output_user data/recbole/ml-1m/ml-1m.user.kg \
  --output_merged data/recbole/ml-1m/ml-1m.kg \
  --item_kg data/recbole/ml-1m/ml-1m.item.kg
```

**输出文件**：
1. `ml-1m.user.kg`: 用户兴趣KG（90,100 triplets）
2. `ml-1m.kg`: 合并KG（122,775 triplets）

**格式**：
```
head_id:token	relation_id:token	tail_id:token
1	long_term_interest	colorful_tones
1	long_term_interest	centered
1	short_term_interest	bright_lighting
...
```

---

## 📊 最终结果统计

### **用户兴趣提取统计**

```
Total users: 6,040
Users with long-term: 6,036 (99.9%)
Users with short-term: 6,010 (99.5%)

Total long-term interests: 30,179
Total short-term interests: 59,921
Total triplets: 90,100
```

**分布**：
- 平均长期兴趣/用户: 5.0
- 平均短期兴趣/用户: 10.0
- 平均LLM调用/用户: 1.0（受ML-1M数据集限制）

### **知识图谱规模**

```
Item KG (ml-1m.item.kg):
  - 3,415 movies
  - 32,675 triplets
  - 15 item relations

User KG (ml-1m.user.kg):
  - 6,040 users
  - 90,100 triplets
  - 2 user relations (long_term_interest, short_term_interest)

Merged KG (ml-1m.kg):
  - Total: 122,775 triplets
  - Ready for KGAT, CKE, RippleNet training
```

---

## 🔧 关键问题与解决方案

### **问题1：LLM API接口不匹配**

**现象**：`'OpenAIMLLM' object has no attribute 'generate'`

**原因**：MLLM类的`generate()`方法用于多模态（图片+文本），不适用纯文本

**解决**：
```python
# ❌ 错误：
response = mllm.generate(text_prompt=prompt, ...)

# ✅ 正确：
messages = [{"role": "user", "content": prompt}]
response = mllm.client.chat.completions.create(
    model=mllm.model_name,
    messages=messages,
    temperature=0.0,
    max_tokens=500
)
```

### **问题2：LLM改写relation/entity**

**现象**：
- `composition_styles` → `composition_style` (去掉复数)
- `bright lighting` → `bright` (简化)

**影响**：词表不匹配，导致KG连接失败

**解决**：强化Prompt规则
```
**CRITICAL RULES:**
1. Use EXACT relation and entity strings
   - DO NOT modify relation names
   - DO NOT simplify entities
   - Copy-paste exactly as shown
```

### **问题3：年龄计算错误**

**现象**：所有新兴趣的age都是168（应该是84）

**原因**：Prompt中JSON示例使用固定数字（168, 84），LLM直接抄

**解决**：
1. 移除固定数字示例
2. 添加明确的计算公式
3. 使用占位符替代固定值

```json
// ❌ Before:
[
  {"relation": "mood", "entity": "romantic", "age_days": 168},
  {"relation": "genre", "entity": "drama", "age_days": 84}
]

// ✅ After:
[
  {"relation": "<exact_relation>", "entity": "<exact_entity>", "age_days": <calculated_number>}
]
```

### **问题4：User KG格式错误**

**现象**：Tail包含多余的relation前缀
```
1  long_term_interest  composition_styles:centered  ❌
```

**期望**：
```
1  long_term_interest  centered  ✅
```

**原因**：转换脚本错误拼接了relation

**解决**：
```python
# ❌ 错误：
triplets.append((user_id, 'long_term_interest', f"{relation}:{entity}"))

# ✅ 正确：
entity = interest['entity']
entity = entity.replace(' ', '_')  # 归一化
triplets.append((user_id, 'long_term_interest', entity))
```

---

## 📂 文件清单

### **核心模块**

1. ✅ `src/extraction/user_interest_extractor.py`
   - `UserInterestExtractor` class
   - Short-term statistical extraction
   - Long-term LLM summarization
   - Helper functions

### **脚本**

2. ✅ `scripts/extract_user_interests_hybrid.py`
   - CLI wrapper
   - Multi-threaded execution
   - Progress tracking

3. ✅ `scripts/convert_user_interests_to_kg.py`
   - JSON → RecBole format
   - User KG generation
   - Item+User KG merging

### **分析工具**

4. ✅ `scripts/check_timestamp_issue.py`
   - 诊断ML-1M时间戳问题
   - 发现59%用户单日批量评分

5. ✅ `scripts/analyze_llm_cost.py`
   - 成本估算
   - 活跃天数统计
   - LLM调用次数预测

6. ✅ `notebooks/analyze_user_interactions.ipynb`
   - 交互式数据分析
   - 可视化

### **输出数据**

7. ✅ `results/user_interests_hybrid.json`
   - 原始提取结果（6,040用户）
   - 包含config、stats、results

8. ✅ `data/recbole/ml-1m/ml-1m.user.kg`
   - 用户兴趣KG（90,100 triplets）

9. ✅ `data/recbole/ml-1m/ml-1m.kg`
   - 合并KG（122,775 triplets）

---

## 🎯 消融实验支持

生成的三个KG文件支持多种消融实验：

### **实验1：Item KG vs User KG vs Full KG**

```bash
# Baseline: No KG
python run_recbole.py --config baseline_no_kg.yaml

# Item KG only
python run_recbole.py --config with_item_kg.yaml \
  --kg_file data/recbole/ml-1m/ml-1m.item.kg

# User KG only
python run_recbole.py --config with_user_kg.yaml \
  --kg_file data/recbole/ml-1m/ml-1m.user.kg

# Full merged KG
python run_recbole.py --config with_full_kg.yaml \
  --kg_file data/recbole/ml-1m/ml-1m.kg
```

### **实验2：Short-term vs Long-term Interests**

修改`convert_user_interests_to_kg.py`，只保留一种relation：
- 只用`long_term_interest`
- 只用`short_term_interest`
- 两者都用（默认）

### **实验3：统计 vs LLM方法**

对比：
- Pure statistical: 只用short-term统计聚合
- Hybrid (current): 统计 + LLM
- Pure LLM: 让LLM同时提取短期和长期

---

## 💰 成本总结

### **Phase 5总成本**

| 项目 | LLM调用 | 模型 | 成本 |
|------|---------|------|------|
| 6,040用户兴趣提取 | 6,056 | gpt-4o-mini | **$6-12** |

### **与其他Phase对比**

| Phase | 任务 | 成本 |
|-------|------|------|
| Phase 1 | 探索（170部） | $0.50 |
| Phase 3 | 验证（800部） | $2.00 |
| Phase 4 | 电影KG（3,415部） | $3.50 |
| **Phase 5** | **用户KG（6,040用户）** | **$6-12** |
| **总计** | **全流程** | **~$15-20** |

**结论**：成本非常合理，完全在预算内

---

## 📈 数据集限制与应对

### **ML-1M数据集已知问题**

1. **时间戳 = 评分提交时间（非观影时间）**
   - 59%用户在注册日批量导入历史评分
   - 导致大量评分集中在同一天

2. **影响**：
   - 中位数活跃天数 = 1天
   - 大多数用户只有1个短期桶
   - 平均LLM调用/用户 = 1.0（退化为整体兴趣）

3. **为什么仍然坚持此方法？**
   - ✅ 方法论正确（适用于更好的数据集）
   - ✅ 论文会在Amazon等数据集上验证（没有此问题）
   - ✅ ML-1M虽退化，但仍能提取有意义的兴趣
   - ✅ 保持方法一致性

### **未来数据集计划**

- **Amazon Reviews**: 真实时间序列，跨度长
- **Last.fm**: 音乐推荐，时序完整
- **Yelp**: 餐厅推荐，地理+时序

---

## ✅ Phase 5完成标志

- ✅ 用户兴趣提取模块实现（统计+LLM）
- ✅ 6,040用户全量提取完成
- ✅ LLM prompt优化（格式匹配、年龄计算）
- ✅ RecBole KG格式转换
- ✅ 三个KG文件生成（item, user, merged）
- ✅ 统计验证（99.9%用户有兴趣）
- ✅ 成本控制（$6-12，符合预期）
- ✅ 代码组织优化（模块化设计）

---

## 🎓 论文贡献点

### **方法论贡献**

1. **Hybrid Statistical+LLM Interest Extraction**
   - 统计聚合捕获短期模式
   - LLM推理识别长期趋势
   - 平衡效率与质量

2. **Temporal Interest Evolution Tracking**
   - 21天短期桶 + 84天长期总结
   - 年龄标记追踪兴趣持续时间
   - 支持推荐模型的时序建模

3. **Knowledge Graph Integration**
   - User interests作为KG entities
   - 与item KG无缝对齐
   - 统一的relation-entity框架

### **实验设计**

- 多层次消融实验（item vs user vs full KG）
- 短期 vs 长期兴趣对比
- 统计 vs LLM方法对比

### **可复现性**

- 完整代码开源
- 详细文档记录
- 成本可控（$15-20全流程）

---

## 📚 相关文档

- `KNOWLEDGE_EXTRACTION_PLAN.md` - 整体计划
- `docs/PHASE3_PROMPT_DESIGN.md` - Phase 3 prompt设计
- `scripts/PHASE4_README.md` - Phase 4电影KG提取
- `src/extraction/user_interest_extractor.py` - 核心实现

---

## 🚦 下一步

1. **Baseline模型训练**
   - RecBole配置文件编写
   - KGAT, CKE, RippleNet训练
   - 性能对比

2. **消融实验**
   - Item KG only
   - User KG only
   - Full KG
   - 短期 vs 长期

3. **论文撰写**
   - Method section
   - Experiment design
   - Results analysis

---

**Phase 5 Status: ✅ 完成！**

*Last updated: 2026-01-12*
