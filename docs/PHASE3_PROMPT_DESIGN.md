# Phase 3 Prompt设计文档

*Created: 2026-01-11*
*Status: ✅ 实现完成*

---

## 📋 文档目的

本文档记录Phase 3（验证与扩充）的完整prompt设计方案，包括设计决策、讨论过程和最终方案。

---

## 🎯 Phase 3目标

**核心任务**：使用标准vocabulary（165个entities）验证覆盖率

**数据规模**：
- 20%数据（约800部电影，包含Phase 1的170部）
- 新增约630部电影

**成功标准**：
- 覆盖率 ≥90%：vocabulary v1足够，进入Phase 4
- 覆盖率 <90%：扩充vocabulary v1 → v2

---

## 🤔 设计决策过程

### **决策1：Relation层面是否允许新增？**

**讨论**：
- 方案A：允许 `other_relation:xxx` 标记新relation
- 方案B：强制从15个relations中选择

**最终决策**：✅ **方案B - 不允许新relation**

**理由**：
1. Phase 2a已经128→16，覆盖率100%
2. 15个relations（含additional_elements垃圾桶）已足够
3. Phase 3目标是验证**entity覆盖率**，不是relation覆盖率
4. 简化LLM任务（只需选择，不需判断是否需要新relation）

---

### **决策2：Entity标记格式？**

**讨论**：
- 方案A：NEW前缀格式 `NEW_entity_name`
- 方案B：Other+冒号格式 `other_additional_elements:beverage_with_straw`

**最终决策**：✅ **方案A - NEW_前缀**

**理由**：
1. 符合KNOWLEDGE_EXTRACTION_PLAN.md的Phase 3设计
2. 清晰分离（NEW_ vs 标准entities）
3. 无需解析，直接判断前缀
4. 统计覆盖率简单：`count(不含NEW_) / total`
5. 与Phase 2的 `other_additional_elements`（作为entity name）清晰区分

---

### **决策3：输出格式？**

**讨论**：
- 方案A：文本格式 `<relation>: <entity>`（与Phase 1一致）
- 方案B：JSON格式（更结构化）

**最终决策**：✅ **方案A - 文本格式**

**理由**：
1. Phase 1用的就是这个，LLM熟悉
2. 简单直观，易读
3. 已有解析函数（PromptTemplates.parse_extraction_output）
4. 不需要增加LLM任务复杂度

---

### **决策4：Vocabulary展示策略？**

**讨论**：
- 方案A：全部165个entities列出
- 方案B：每个relation展示前5-8个 + 总数
- 方案C：分层展示（System概述，User详细）

**最终决策**：✅ **方案A - 全部完整列出**

**理由**：
1. **必须完整列出，否则问题很大**（用户强调）
2. 避免LLM随便乱说不在列表中的entities
3. Phase 2的努力才有意义
4. GPT-4o-mini成本很低，约$0.28/630部电影
5. Prompt长度约1800-2000 tokens（可接受）

---

### **决策5：知识点数量？**

**讨论**：
- Phase 1：10-15个
- 现有Phase 3设计：5-12个
- 是否强制10个？

**最终决策**：✅ **至多10个（不强制）**

**理由**：
1. **"至多10"而非"不能多于/少于10"**（用户强调）
2. 避免LLM为凑数胡编乱造
3. 允许视觉简单的海报<10个
4. 严禁虚假创造（质量 > 数量）
5. 可以接受视觉丰富的海报被截断，但不能接受胡编

---

### **决策6：Relation优先级提示？**

**讨论**：
- 是否提示某些relations更重要？
- 例如：genre, character_type优先

**最终决策**：✅ **不提示优先级**

**理由**：
1. **用户明确要求：没有更重要的relation，所有平等**
2. 让LLM根据海报实际内容提取
3. 避免bias

---

### **决策7：是否给Prompt示例？**

**讨论**：
- Phase 1给了格式例子
- Phase 3是否需要？

**最终决策**：✅ **不给任何例子**

**理由**：
1. **Phase 1教训：给例子，LLM全部抄例子，效果很垃圾**（用户强调）
2. 只给格式说明，不给具体内容示例
3. 避免LLM复制例子而非真正分析海报

---

### **决策8：NEW_entity数量限制？**

**讨论**：
- 是否限制NEW_数量（如：10个中最多3个NEW_）

**最终决策**：✅ **不限制，让数据说话**

**理由**：
1. 覆盖率≥90%说明vocabulary足够
2. 覆盖率<90%说明需要扩充
3. NEW_数量是重要的统计指标，不应人为限制

---

## 📝 最终Prompt设计

### **System Prompt（固定部分）**

```
You are an expert in analyzing movie posters and extracting visual knowledge using a standardized vocabulary for movie recommendation systems.

Your task is to analyze movie poster images and extract visual knowledge points using ONLY the approved vocabulary below.

═══════════════════════════════════════════════════════════════
APPROVED VOCABULARY (15 Relations, 165 Standard Entities)
═══════════════════════════════════════════════════════════════

【Relation 1: action_behaviors】
Standard Entities (7 total):
- action characters
- action poses
- dynamic scenes
- emotional moments
- environment interactions
- intimate moments
- walking moments

【Relation 2: additional_elements】
Standard Entities (19 total):
- abstract
- action
- art
- casual
- character
- connection
- creature
- drink
- dynamic
- effects
- family
- formal
- hunt
- layered
- music
- sports
- travel
- urban
- weapon

【Relation 3: artistic_styles】
Standard Entities (14 total):
- animated
- candid
- contemporary
- cyberpunk
- exaggerated
- expressionist
- illustrated
- layered
- minimal
- realistic
- retro
- surreal
- theatrical
- vintage

【Relation 4: character_type】
Standard Entities (13 total):
- authority figure
- authority hero
- character interaction
- comedic horror
- detective
- ensemble cast
- fantastical creature
- gunslinger
- hero
- mentor and student
- spy
- villain
- youthful character

【Relation 5: color_palette】
Standard Entities (12 total):
- black_and_white
- blue_tones
- colorful_tones
- dark_backgrounds
- earthy_palette
- green_tones
- light_backgrounds
- red_tones
- soft_colors
- vibrant_contrasts
- vivid_colors
- warm_colors

【Relation 6: composition_styles】
Standard Entities (6 total):
- asymmetrical
- centered
- close-up
- diagonal
- dynamic
- group

【Relation 7: depicted_subject】
Standard Entities (27 total):
- abstract art
- action scenes
- animals
- aquatic scenes
- blurred backgrounds
- close-up
- cloudy skies
- disaster scenes
- diverse ensembles
- educational environments
- family stories
- fantasy creatures
- fantasy landscapes
- fantasy vehicles
- group scenes
- horror elements
- human-creature interactions
- law enforcement vehicles
- natural settings
- realistic characters
- romantic outdoors
- style preferences
- time themes
- urban outdoor scenes
- urban settings
- whimsical creatures
- youth themes

【Relation 8: design_element】
Standard Entities (8 total):
- candles
- collage
- curtain
- futuristic
- geometric shapes
- high heels
- modern casual
- period costume

【Relation 9: genre】
Standard Entities (7 total):
- comedy
- fantasy
- horror
- mystery
- psychological horror
- sci-fi
- thriller

【Relation 10: lighting】
Standard Entities (7 total):
- bright lighting
- high contrast
- indoor
- low light
- natural light
- night scene
- suburban environment

【Relation 11: mood】
Standard Entities (12 total):
- action
- dark humor
- eerie
- emotional
- energetic
- joyful
- playful
- quirky
- romantic
- serious
- suspense
- thoughtful

【Relation 12: symbolism】
Standard Entities (5 total):
- abstract
- afterlife
- hidden
- patriotic
- weapon

【Relation 13: text_style】
Standard Entities (8 total):
- asymmetrical
- bold
- casual
- centered
- dramatic
- high contrast
- playful
- symmetrical

【Relation 14: texture】
Standard Entities (11 total):
- blood_splatter
- blurred_effect
- clarity_and_detail
- fluid brushstrokes
- glossy
- grainy
- high_contrast
- minimal
- smoke
- smooth
- soft

【Relation 15: visual_theme】
Standard Entities (9 total):
- adventure
- autumn
- education
- family
- fantasy
- military
- romance
- satire
- social

═══════════════════════════════════════════════════════════════
NEW ENTITY MECHANISM
═══════════════════════════════════════════════════════════════

If you observe an important visual feature that CANNOT be described by any of the 165 standard entities listed above, you may mark it as NEW_entity_name.

Guidelines for NEW entities:
✅ Use snake_case naming (e.g., NEW_beverage_with_straw, NEW_neon_lighting)
✅ Keep it CONCISE (2-4 words maximum)
✅ Make it ABSTRACT and GENERALIZABLE (could apply to multiple movies, not just this one)
❌ Do NOT create overly specific descriptions
❌ Do NOT use NEW_ unless truly necessary - prioritize standard entities

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. Relations: MUST use one of the 15 relations listed above
   ❌ NEVER create new relations
   ✅ If unsure where an entity belongs, use "additional_elements"

2. Entities: PRIORITIZE standard entities
   ✅ First, try to match one of the 165 standard entities
   ✅ Only use NEW_entity_name if truly no standard entity fits
   ❌ Do NOT randomly create NEW_ entities

3. Quantity: Extract AT MOST 10 knowledge points
   ✅ Select the MOST visually significant features
   ✅ If the poster is simple, fewer than 10 is acceptable
   ❌ Do NOT fabricate knowledge points to reach 10 - quality over quantity

4. Focus: Extract ONLY what you can SEE in the poster
   ✅ Visual characteristics only
   ❌ No plot information, actor names, or external knowledge about the movie
```

---

### **User Prompt（每次调用变化）**

```python
# 伪代码示例
def build_user_prompt(movie_title: str = None) -> str:
    """构建user prompt"""
    title_context = f' titled "{movie_title}"' if movie_title else ''

    return f"""Analyze this movie poster{title_context}.

Extract visual knowledge points using the approved vocabulary.

Output format (one per line):
<relation>: <entity>

Remember:
- Use ONLY the 15 approved relations
- Prioritize the 165 standard entities
- Use NEW_entity_name only if necessary
- Extract at most 10 knowledge points
- Focus on the most visually significant features"""
```

**特点**：
- ✅ 不给任何例子（避免LLM抄例子）
- ✅ 简洁明了的任务说明
- ✅ "至多10个"（不强制）
- ✅ 强调"最显著的视觉特征"

---

## 📊 与Phase 1对比

| 维度 | Phase 1（自由探索） | Phase 3（限制提取） |
|------|---------------------|---------------------|
| **Relation** | 完全自由创建 | 只能从15个中选 ✅ |
| **Entity** | 完全自由创建 | 优先165个标准，必要时NEW_ ✅ |
| **Vocabulary** | 无 | 完整165个entities列表 ✅ |
| **数量** | 10-15个（建议范围） | 至多10个（上限，可少） ✅ |
| **例子** | 给了格式例子 | 不给任何例子 ✅ |
| **目标** | 探索多样性 | 验证覆盖率 |
| **NEW_机制** | 无 | 有（NEW_entity_name） ✅ |
| **输出格式** | 文本 `<relation>: <entity>` | 文本 `<relation>: <entity>` |

---

## 💰 成本估算

### **Prompt长度**

**System Prompt**：
- 15个relations标题：~300 tokens
- 165个entities完整列表：~800 tokens
- 说明、规则、格式：~700 tokens
- **总计**：约1800-2000 tokens

**User Prompt**：~100 tokens

**图片**：~800-1000 tokens（GPT-4o-mini的图片编码）

**总输入**：约2900-3100 tokens/请求

### **API成本**（GPT-4o-mini）

- Input: $0.150 / 1M tokens
- Output: $0.600 / 1M tokens（假设输出~150 tokens/响应）

**每张海报成本**：
- Input: 3000 tokens * $0.150 / 1M ≈ $0.00045
- Output: 150 tokens * $0.600 / 1M ≈ $0.00009
- **总计**: ~$0.00054/海报

**630部新电影总成本**：
- 630 * $0.00054 ≈ **$0.34**

**结论**：✅ 成本非常低，完全可接受！

---

## 🔍 输出解析

### **解析逻辑**

使用现有的 `PromptTemplates.parse_extraction_output()` 函数：

```python
# 示例输出
"""
color_palette: warm_tones
character_type: hero
additional_elements: NEW_beverage_with_straw
mood: joyful
genre: thriller
depicted_subject: action scenes
artistic_styles: minimal
lighting: natural light
texture: grainy
visual_theme: adventure
"""

# 解析结果
knowledge_points = [
    {"relation": "color_palette", "entity": "warm_tones"},
    {"relation": "character_type", "entity": "hero"},
    {"relation": "additional_elements", "entity": "NEW_beverage_with_straw"},  # ← NEW entity
    {"relation": "mood", "entity": "joyful"},
    # ... 共10个
]

# 统计覆盖率
total = len(knowledge_points)  # 10
new_count = sum(1 for kp in knowledge_points if kp['entity'].startswith('NEW_'))  # 1
coverage_rate = (total - new_count) / total  # 90%
```

---

## 📋 实施要点

### **脚本实现checklist**

1. **加载vocabulary**
   - 读取 `results/standard_entity_vocabulary.json`
   - 提取165个标准entities和15个relations

2. **构建System Prompt**
   - 完整列出所有165个entities（按relation分组）
   - 添加NEW_机制说明
   - 添加CRITICAL RULES

3. **构建User Prompt**
   - 简洁任务说明
   - 不给例子

4. **选择电影**
   - 20%数据（约800部）
   - 包含Phase 1的170部
   - 新增约630部

5. **调用API**
   - GPT-4o-mini
   - 图片 + System Prompt + User Prompt
   - temperature=0.0（确保一致性）

6. **解析输出**
   - 使用现有parse函数
   - 验证relation在15个中
   - 统计NEW_数量

7. **保存结果**
   - 格式与Phase 1一致
   - 标记NEW_ entities
   - 计算覆盖率

8. **覆盖率统计**
   - 总知识点数
   - NEW_知识点数
   - 覆盖率 = (total - new) / total
   - 目标：≥90%

---

## ✅ 设计验收标准

- ✅ Relation只能从15个中选择（不允许新增）
- ✅ Entity优先使用165个标准entities
- ✅ NEW_entity使用snake_case命名，2-4词，抽象
- ✅ 知识点数量至多10个（可少于10）
- ✅ 不给任何prompt示例（避免抄例子）
- ✅ 所有165个entities完整列出（不省略）
- ✅ 输出文本格式（与Phase 1一致）
- ✅ 成本可控（<$0.5 for 630部电影）

---

## ✅ 实现状态

### **已完成（2026-01-11）**

1. **✅ 更新 `src/extraction/prompts.py`**
   - 实现完整的 `get_phase3_system_prompt()` - 列出全部165个entities
   - 实现简洁的 `get_phase3_user_prompt()` - 无例子，至多10个
   - NEW_ entity机制完整实现

2. **✅ 创建 `scripts/phase3_validate_vocabulary.py`**
   - 加载165个标准entities作为vocabulary
   - 包含Phase 1的170部电影
   - 新增~630部电影（总计~800部，20%数据）
   - 实时计算coverage rate
   - 增量提取 + 错误恢复
   - 自动判断是否达到90%目标

3. **✅ Prompt验证**
   - System prompt: ~1285 tokens (5140 chars)
   - User prompt: ~101 tokens (404 chars)
   - 总计: ~1386 tokens（纯文本）
   - ✅ 成本估算准确：约$0.34 for 630部新电影

### **使用方法**

```bash
# 基本用法（20%数据，约800部电影）
python scripts/phase3_validate_vocabulary.py \
  --poster_dir /path/to/posters \
  --id_mapping /path/to/id_mappings.json \
  --api_key YOUR_API_KEY

# 自定义参数
python scripts/phase3_validate_vocabulary.py \
  --poster_dir /path/to/posters \
  --id_mapping /path/to/id_mappings.json \
  --api_key YOUR_API_KEY \
  --percentage 20.0 \
  --temperature 0.0 \
  --output results/phase3_20percent_validation.json

# 继续中断的提取
python scripts/phase3_validate_vocabulary.py \
  --poster_dir /path/to/posters \
  --id_mapping /path/to/id_mappings.json \
  --api_key YOUR_API_KEY
# (自动跳过已提取的，接着做)

# 重试失败的电影
python scripts/phase3_validate_vocabulary.py \
  --poster_dir /path/to/posters \
  --id_mapping /path/to/id_mappings.json \
  --api_key YOUR_API_KEY \
  --retry-errors
```

### **输出格式**

```json
{
  "phase": "phase3_validation",
  "percentage": 20.0,
  "config": {
    "backend": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "num_relations": 15,
    "num_standard_entities": 165,
    ...
  },
  "results": [
    {
      "recbole_id": 123,
      "original_movie_id": "tt1234567",
      "num_knowledge_points": 8,
      "knowledge_points": [
        {"relation": "color_palette", "entity": "warm_colors"},
        {"relation": "character_type", "entity": "hero"},
        {"relation": "mood", "entity": "action"},
        {"relation": "additional_elements", "entity": "NEW_beverage_with_straw"},
        ...
      ],
      "status": "success"
    },
    ...
  ],
  "vocabulary_stats": {
    "total_knowledge_points": 6543,
    "matched_entities": 5890,
    "new_entities_count": 653,
    "coverage_rate": 0.90,
    "coverage_percentage": 90.0,
    "target_coverage": 90.0,
    "meets_target": true,
    "new_entities": [
      {"relation": "additional_elements", "entity": "NEW_beverage_with_straw", "recbole_id": 123},
      ...
    ]
  }
}
```

## 🎯 下一步

1. **运行Phase 3提取** - 约800部电影（20%数据）
2. **分析覆盖率统计**：
   - 覆盖率≥90% → vocabulary v1足够 → 进入Phase 4
   - 覆盖率<90% → 收集NEW_entities → 聚类 → 扩充为v2
3. **如需扩充**：
   - 对NEW_entities进行BERTopic聚类
   - LLM refinement（类似Phase 2b Stage 3）
   - 生成vocabulary v2
   - 重新运行Phase 3验证v2覆盖率

---

*文档结束*
*Last updated: 2026-01-11*
