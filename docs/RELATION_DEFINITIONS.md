# Relation职责定义与边界规则

*Created: 2026-01-04*
*Purpose: 为Phase 2b Entity重分配提供明确的relation边界定义*

---

## 📋 概述

本文档定义了**14个有意义的标准relations + 1个杂项relation**的职责边界，用于指导LLM进行entity重分配。

**核心原则**：
- 每个relation必须有明确的、互不重叠的职责范围
- 边界模糊cases需要明确的判断规则
- 优先语义清晰，避免"大杂烩"relations

**使用场景**：
1. Phase 2b-0: Entity重分配（两阶段LLM）
2. Phase 3: 全量提取时的prompt设计
3. 人工审核entity归属时的参考标准

---

## 🎯 14个有意义的Relations

### **1. depicted_subject**

**职责**：画面的主要视觉主体（what is shown）

**范围**：
- ✅ 人物基础类型：man, woman, child, elderly_person, face, human_figure, human_portrait
- ✅ 非人物对象：building, car, weapon, statue, animal, dog, cat, vehicle
- ✅ 场景：landscape, cityscape, ocean, sky, mountain, forest, urban_scene
- ❌ 人物的角色/职业/关系 → **character_type**

**边界规则**：
```
IF entity是人物相关:
    IF 有明确角色/职业/关系属性（warrior, detective, couple）:
        → character_type
    ELSE (仅基础人物描述，如man, woman, child):
        → depicted_subject
ELSE (非人物):
    → depicted_subject
```

**示例**：
| Entity | 判断 | 原因 |
|--------|------|------|
| man | ✅ depicted_subject | 基础人物类型 |
| warrior | ❌ → character_type | 角色职业 |
| lone_man | ✅ depicted_subject | 只描述数量，无角色属性 |
| landscape | ✅ depicted_subject | 非人物场景 |
| building | ✅ depicted_subject | 非人物对象 |

**备注**：
- 本relation在2026-01-04合并了原`depicted_entities`（6个entities）
- 合并理由：depicted_entities数据量太少且语义重叠

---

### **2. character_type**

**职责**：人物的角色类型、职业、身份、关系（**仅限人物相关**）

**范围**：
- ✅ 职业/角色：warrior, detective, hero, villain, cowboy, spy, soldier, scientist
- ✅ 关系：couple, romantic_couple, family, group, lovers, friends
- ✅ 社会身份：outlaw, royalty, civilian, rebel, leader
- ✅ 角色属性：protagonist, antagonist, sidekick
- ❌ 基础人物类型（man, woman, child） → **depicted_subject**
- ❌ 非人物 → **depicted_subject**

**边界规则**：
```
IF entity明确表达人物的角色/职业/关系/社会身份:
    → character_type
ELSE IF entity仅描述人物的基础类型/数量/性别:
    → depicted_subject
```

**示例**：
| Entity | 判断 | 原因 |
|--------|------|------|
| warrior | ✅ character_type | 角色职业 |
| man | ❌ → depicted_subject | 基础人物类型 |
| single_man | ❌ → depicted_subject | 只描述数量 |
| man_in_suit | ✅ character_type | 暗示身份（商人/特工） |
| couple | ✅ character_type | 关系 |
| detective | ✅ character_type | 职业 |

**模糊cases**：
- `lone_man` → depicted_subject（只描述数量，无角色属性）
- `armed_man` → character_type（"armed"暗示角色：士兵/罪犯）
- `young_woman` → depicted_subject（只描述年龄性别）
- `rebellious_teenager` → character_type（"rebellious"暗示角色属性）

---

### **3. visual_theme**

**职责**：整体视觉/叙事主题、概念性主题（**不包括电影类型**）

**范围**：
- ✅ 叙事主题：war, romance, adventure, survival, coming_of_age, betrayal, revenge, redemption
- ✅ 概念主题：urban_life, nature, technology, isolation, freedom, tradition, modernity
- ✅ 视觉概念：minimalism, surrealism, realism, abstraction, symmetry_theme, chaos
- ❌ 单纯颜色 → **color_palette**
- ❌ 单纯情绪 → **mood**
- ❌ 艺术流派 → **artistic_styles**
- ❌ 电影类型（horror, thriller, comedy, action, sci-fi等） → **genre**

**核心判断标准**：
> 问自己："这是在描述**什么主题/讲什么故事的内容**，还是在描述**什么类型的电影/什么颜色/什么情绪/什么风格**？"
> - 如果是主题/故事内容 → visual_theme
> - 如果是电影类型 → genre
> - 如果是颜色/情绪/风格 → 对应的专门relation

**主题 vs 类型的区别**：
- **主题（theme）**：描述故事讲什么、传达什么概念（war战争, betrayal背叛, freedom自由）
- **类型（genre）**：描述电影属于什么类别（horror恐怖片, thriller惊悚片, comedy喜剧片）
- 例子：
  - "war" = 战争主题 → visual_theme
  - "war_film" = 战争片类型 → genre
  - "horror" = 恐怖片类型 → genre（不是"恐怖主题"）
  - "psychological_thriller" = 心理惊悚片类型 → genre（不是"心理主题"）

**边界规则（最复杂，需仔细判断）**：
| Entity | 判断 | 原因 |
|--------|------|------|
| "war" | ✅ visual_theme | 叙事主题（战争） |
| "romance" | ✅ visual_theme | 叙事主题（浪漫） |
| "adventure" | ✅ visual_theme | 叙事主题（冒险） |
| "betrayal" | ✅ visual_theme | 叙事主题（背叛） |
| "freedom" | ✅ visual_theme | 概念主题（自由） |
| "urban_life" | ✅ visual_theme | 概念主题（城市生活） |
| "minimalism" | ✅ visual_theme | 视觉概念（极简主义） |
| **--- 以下是genre，不是theme ---** |
| "horror" | ❌ → genre | 恐怖**片类型**，不是主题 |
| "thriller" | ❌ → genre | 惊悚**片类型**，不是主题 |
| "psychological_thriller" | ❌ → genre | 心理惊悚**片类型** |
| "crime_thriller" | ❌ → genre | 犯罪惊悚**片类型** |
| "action" | ❌ → genre | 动作**片类型** |
| "comedy" | ❌ → genre | 喜剧**片类型** |
| "horror_comedy" | ❌ → genre | 恐怖喜剧**片类型** |
| "sci-fi" | ❌ → genre | 科幻**片类型** |
| **--- 其他排除情况 ---** |
| "red" | ❌ → color_palette | 单纯颜色 |
| "romantic" | ❌ → mood | 情绪形容词 |
| "suspenseful" | ❌ → mood | 情绪形容词 |
| "dark" | ❌ → mood | 情绪描述 |
| "noir" | ❌ → artistic_styles | 艺术流派 |
| "minimalist" | ❌ → artistic_styles | 风格形容词 |

**命名规则辅助判断**：
- 如果是形容词形式（romantic, dark, minimalist） → 倾向mood或artistic_styles
- 如果是名词形式（romance, darkness, minimalism） → 倾向visual_theme（作为概念）
- 例外：如果名词明确是电影类型词（horror, thriller, comedy），即使是名词也要→ genre

**关于surrealism/minimalism等视觉概念词**：
- `surrealism`, `minimalism`, `realism` 等名词 → visual_theme（作为视觉概念/主题）
- `surrealist`, `minimalist`, `realist` 等形容词 → artistic_styles（作为风格）
- `surrealist_style`, `minimalist_style` → artistic_styles（明确标注为风格）

**为什么保留visual_theme？**
- 有些概念性主题无法归入其他relations（如war, urban_life, freedom）
- "主题"是海报传达的核心叙事/概念，区别于表面的颜色/情绪/风格
- 例：一张战争片海报，visual_theme=war，mood=tense，color_palette=desaturated_colors

---

### **4. color_palette**

**职责**：颜色及配色方案

**范围**：
- ✅ 单一颜色：red, blue, green, black, white, yellow, orange, purple
- ✅ 颜色组合：black_and_white, red_and_blue, tricolor
- ✅ 配色方案：warm_tones, cool_colors, monochromatic, vibrant_palette, muted_colors, pastel_palette
- ✅ 色调描述：pastel, sepia, saturated, desaturated, high_saturation
- ❌ 情绪性的"dark"（作为情绪） → **mood**

**边界规则**：
```
IF entity描述颜色或颜色组合:
    → color_palette
ELSE IF entity是情绪词（碰巧有颜色含义，如"dark"表示阴暗情绪）:
    → mood
```

**示例**：
| Entity | 判断 | 原因 |
|--------|------|------|
| red | ✅ color_palette | 单一颜色 |
| red_theme | ✅ color_palette | 强调颜色主题 |
| warm_tones | ✅ color_palette | 配色方案 |
| dark | ⚠️ 模糊 | 如果指颜色 → color_palette，如果指情绪 → mood |
| dark_colors | ✅ color_palette | 明确指颜色 |
| monochrome | ✅ color_palette | 配色方案 |

**用户决策（2026-01-04）**：
- 单一颜色也算color_palette ✅
- 理由：单一颜色放visual_theme更奇怪

---

### **5. mood**

**职责**：情绪、氛围、感受（形容词性）

**范围**：
- ✅ 情绪形容词：dramatic, romantic, melancholic, tense, joyful, mysterious, dark, uplifting, ominous, hopeful
- ✅ 氛围描述：suspenseful, serene, chaotic, peaceful, intense, playful, somber
- ❌ 叙事主题（romance, war） → **visual_theme**
- ❌ 颜色（dark作为颜色） → **color_palette**

**边界规则**：
```
IF entity是形容词 AND 描述情绪/感受/氛围:
    → mood
ELSE IF entity是名词 AND 是叙事主题:
    → visual_theme
```

**示例**：
| Entity | 判断 | 原因 |
|--------|------|------|
| romantic | ✅ mood | 情绪形容词 |
| romance | ❌ → visual_theme | 叙事主题（名词） |
| dark | ✅ mood | 情绪（阴暗感） |
| darkness_theme | ❌ → visual_theme | 概念主题 |
| mysterious | ✅ mood | 情绪 |
| tense | ✅ mood | 情绪 |

---

### **6. artistic_styles**

**职责**：艺术流派、风格流派、视觉风格技法

**范围**：
- ✅ 艺术流派：art_deco, impressionistic, expressionism, cubism, surrealist_style
- ✅ 时代风格：vintage, retro, modern, contemporary, classic, futuristic_style
- ✅ 电影风格：film_noir, neo_noir, grunge, gothic_style
- ✅ 设计风格：minimalist_style, maximalist, ornate, clean_design
- ❌ 物理质感（grainy, high_contrast） → **texture**
- ❌ 视觉概念（minimalism作为概念） → **visual_theme**

**边界规则**：
```
IF entity描述艺术流派/风格流派:
    → artistic_styles
ELSE IF entity描述物理质感（颗粒、对比度）:
    → texture
ELSE IF entity描述整体视觉概念（作为主题）:
    → visual_theme
```

**示例**：
| Entity | 判断 | 原因 |
|--------|------|------|
| vintage | ✅ artistic_styles | 复古风格 |
| vintage_look | ❌ → texture | 物理褪色效果 |
| minimalist | ✅ artistic_styles | 风格形容词 |
| minimalism | ❌ → visual_theme | 视觉概念（名词） |
| noir | ✅ artistic_styles | 风格流派 |
| art_deco | ✅ artistic_styles | 艺术流派 |

**命名规则辅助**：
- `xxx_style` → artistic_styles（如vintage_style, gothic_style）
- `xxx_look` → texture（如vintage_look表示物理效果）
- `xxx` (裸词) → 需结合语义判断

---

### **7. texture**

**职责**：物理/技术层面的视觉质感

**范围**：
- ✅ 颗粒质感：grainy, film_grain, smooth, rough, sandy_texture
- ✅ 对比度：high_contrast, low_contrast, contrasty, soft_contrast
- ✅ 焦距效果：soft_focus, sharp, shallow_depth_of_field, blurred_background, bokeh
- ✅ 物理状态：faded, weathered, crisp, glossy, matte, worn, distressed
- ❌ 艺术风格 → **artistic_styles**

**边界规则**：
```
IF entity描述物理/技术参数（颗粒、对比度、焦距、表面状态）:
    → texture
ELSE IF entity描述艺术风格/流派:
    → artistic_styles
```

**示例**：
| Entity | 判断 | 原因 |
|--------|------|------|
| grainy | ✅ texture | 颗粒质感 |
| high_contrast | ✅ texture | 对比度 |
| soft_focus | ✅ texture | 焦距效果 |
| vintage_look | ✅ texture | 物理褪色效果 |
| vintage_style | ❌ → artistic_styles | 艺术风格 |
| film_grain | ✅ texture | 胶片颗粒 |

---

### **8. genre**

**职责**：电影类型

**范围**：
- ✅ 主流类型：action, romance, thriller, horror, comedy, drama, sci-fi, fantasy, western
- ✅ 细分类型：romantic_comedy, action_thriller, psychological_horror
- ❌ 视觉主题 → **visual_theme**

**边界规则**：
```
IF entity是标准电影类型分类:
    → genre
ELSE:
    判断是否属于其他relations
```

**示例**：
| Entity | 判断 |
|--------|------|
| action | ✅ genre |
| romance | ⚠️ 模糊（可能是genre或visual_theme，需context） |
| thriller | ✅ genre |
| war | ❌ → visual_theme（战争是主题，不是类型；war_film才是类型） |

**备注**：
- genre是最清晰的relation之一，边界明确
- 优先保留所有genre entities（电影类型是推荐核心特征）

---

### **9. lighting**

**职责**：光照条件和打光技术

**范围**：
- ✅ 光源类型：natural_light, artificial_light, candlelight, firelight, neon_lighting
- ✅ 打光技术：dramatic_lighting, backlit, front_lit, side_lighting, rim_lighting
- ✅ 光照强度：high-key, low-key, bright_lighting, dim_lighting
- ✅ 光照效果：soft_lighting, harsh_lighting, diffused_light, spotlight

**示例**：
| Entity | 判断 |
|--------|------|
| natural_light | ✅ lighting |
| dramatic_lighting | ✅ lighting |
| backlit | ✅ lighting |
| high-key | ✅ lighting |

**备注**：
- lighting是边界非常清晰的relation
- 与texture的区别：lighting是光照条件，texture是视觉质感

---

### **10. text_style**

**职责**：文字排版样式

**范围**：
- ✅ 字体样式：bold_text, serif_font, sans_serif, script_font, handwritten_text
- ✅ 排版布局：large_title, minimal_text, centered_text, vertical_text
- ✅ 文字效果：embossed_text, shadowed_text, outlined_text

**示例**：
| Entity | 判断 |
|--------|------|
| bold_text | ✅ text_style |
| serif_font | ✅ text_style |
| large_title | ✅ text_style |
| minimal_text | ✅ text_style |

**备注**：
- text_style专注于海报上的文字元素
- 与design_element的区别：text_style是文字，design_element是图形元素

---

### **11. symbolism**

**职责**：具有象征意义的符号和图案

**范围**：
- ✅ 宗教符号：cross, star_of_david, crescent_moon, lotus, mandala
- ✅ 国家符号：american_flag, eagle, national_emblem
- ✅ 文化符号：rose, skull, dove, phoenix, dragon
- ✅ 抽象符号：infinity_symbol, yin_yang, hourglass

**示例**：
| Entity | 判断 |
|--------|------|
| cross | ✅ symbolism |
| rose | ✅ symbolism |
| american_flag | ✅ symbolism |
| skull | ✅ symbolism |

**备注**：
- symbolism的entities都有明确的象征意义
- 与depicted_subject的区别：symbolism强调象征意义，depicted_subject强调视觉主体本身

---

### **12. composition_styles**

**职责**：构图技法和画面布局方式

**范围**：
- ✅ 构图法则：rule_of_thirds, golden_ratio, symmetry, asymmetry
- ✅ 布局方式：centered_composition, off-center, diagonal_composition
- ✅ 取景方式：close-up, wide_shot, medium_shot, bird's_eye_view, low_angle

**示例**：
| Entity | 判断 |
|--------|------|
| rule_of_thirds | ✅ composition_styles |
| symmetry | ✅ composition_styles |
| centered_composition | ✅ composition_styles |
| close-up | ✅ composition_styles |

**备注**：
- 与design_element的区别：composition_styles是布局方式，design_element是设计元素本身

---

### **13. design_element**

**职责**：图形设计元素

**范围**：
- ✅ 图形元素：typography, geometric_shapes, border, frame, logo, badge, ribbon
- ✅ 装饰元素：ornament, pattern, texture_overlay
- ❌ 文字样式 → **text_style**
- ❌ 构图布局 → **composition_styles**

**示例**：
| Entity | 判断 |
|--------|------|
| typography | ✅ design_element |
| geometric_shapes | ✅ design_element |
| border | ✅ design_element |
| frame | ✅ design_element |

**备界**：
- design_element是具体的设计图形元素
- 与composition_styles的区别：后者是布局方式，前者是元素本身

---

### **14. action_behaviors**

**职责**：物理动作和行为（动词性）

**范围**：
- ✅ 身体动作：climbing, running, fighting, walking, jumping, dancing, embracing
- ✅ 行为动作：action_scene, dynamic_movement, violent_action, chase

**示例**：
| Entity | 判断 |
|--------|------|
| running | ✅ action_behaviors |
| climbing | ✅ action_behaviors |
| fighting | ✅ action_behaviors |
| dynamic_movement | ✅ action_behaviors |

**备注**：
- 当前数据量很少（6个entities，全部singleton）
- 暂时保留，观察重分配后的情况
- 可能在Phase 3全量提取时会增加

---

## 🗑️ 1个杂项Relation

### **15. additional_elements**

**职责**：无法归类到任何标准relation的其他元素

**范围**：
- ✅ 任何无法归入上述14个relations的entities
- ✅ 边缘cases、混合概念、特殊情况

**示例（当前9个entities）**：
- ocean_wave, night_sky, camera_angle, 1990s, weapon, suit, belt, centered

**使用原则**：
- 这是最后的兜底选项
- 只有在确实无法归入任何其他relation时才使用
- 预期会收集一些长尾、低频、混杂的entities

**备注**：
- 原名`others_relation`，在Phase 2a LLM微调时改名为`additional_elements`
- 作为"others"类别保留，不需要额外创建新的others

---

## 🔄 合并决策记录

### **depicted_entities → depicted_subject** ✅

**合并时间**：2026-01-04（决策阶段）

**理由**：
1. depicted_entities只有6个entities，100% singletons
2. 语义与depicted_subject高度重叠（都是描述画面主体）
3. LLM在提取时很少使用这个relation

**合并后的depicted_subject**：
- 包含：人物基础类型 + 物体 + 场景 + 动物（原depicted_entities）

---

## 📊 使用指南

### **For LLM Prompt设计**

在第一轮entity清理时，为每个relation提供：
1. 职责定义（1句话）
2. 范围描述（✅包含什么，❌排除什么）
3. 边界规则（IF-ELSE逻辑）
4. 2-3个示例

示例prompt模板：
```
Relation: visual_theme

职责: 整体视觉/叙事主题、概念性主题

包含:
- 叙事主题（war, romance, adventure）
- 概念主题（urban_life, freedom, isolation）
- 视觉概念（minimalism, surrealism, realism）

排除:
- 单纯颜色 → color_palette
- 单纯情绪 → mood
- 艺术流派 → artistic_styles

判断规则:
问自己："这是在描述什么主题/讲什么故事，还是在描述什么颜色/情绪/风格？"
- 如果是主题/故事 → visual_theme
- 否则 → 对应的专门relation

示例:
✅ "romance" (叙事主题)
✅ "war" (叙事主题)
✅ "minimalism" (视觉概念)
❌ "romantic" → mood (情绪)
❌ "red" → color_palette (颜色)
❌ "vintage" → artistic_styles (风格)
```

### **For 人工审核**

审核优先级：
1. **高频entities的变动**（频数>10）- 重点关注
2. **有争议的边界cases**（如romantic vs romance）
3. **被移除到additional_elements的entities** - 确认是否合理

审核问题：
- 这个entity被分配到的relation是否语义一致？
- 是否有更合适的relation？
- 如果是被移除的，移除原因是否合理？

---

## 📝 更新日志

### 2026-01-04
- ✅ 创建本文档
- ✅ 定义14+1个relations的完整职责和边界
- ✅ 明确visual_theme, depicted_subject vs character_type的复杂边界规则
- ✅ 记录depicted_entities → depicted_subject的合并决策

---

*文档结束*
