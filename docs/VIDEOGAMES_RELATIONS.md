# Video Games Relation Vocabulary v1.0

基于Phase 1探索结果（748个成功样本，4220个知识点）分析得出的14个核心relations + 1个additional。

## 📊 统计摘要

- **Phase 1样本**: 748个游戏成功提取
- **知识点总数**: 4,220个
- **原始relations**: 291个唯一值
- **归一化后**: 200个语义组
- **核心relations**: 14 + 1 additional

---

## 🎮 核心Relations定义表

| # | Relation | 中文描述 | 覆盖范围 | Phase 1占比 |
|---|----------|----------|----------|------------|
| 1 | `has_visual_style` | 视觉风格 | 写实/卡通/像素/动漫等艺术风格 | 14.88% |
| 2 | `features_character` | 角色特征 | 角色设计、服装、装备、外观 | 17.68% |
| 3 | `set_in_environment` | 环境设定 | 城市/奇幻/科幻/自然等场景 | 14.43% |
| 4 | `has_color_palette` | 色彩搭配 | 主色调、配色方案 | 14.69% |
| 5 | `shows_perspective` | 视角透视 | 第一人称/第三人称/俯视/横版 | 9.57% |
| 6 | `has_ui_elements` | 界面元素 | HUD、菜单、按钮等UI组件 | 7.32% |
| 7 | `has_atmosphere` | 氛围基调 | 黑暗/明亮/严肃/轻松等氛围 | 5.83% |
| 8 | `has_graphics_quality` | 画面质量 | 画质等级、图形保真度 | 0.36% |
| 9 | `features_vehicle` | 载具交通 | 车辆、坐骑、交通工具 | 0.36% |
| 10 | `features_weapon` | 武器装备 | 武器、战斗装备、军械 | - |
| 11 | `has_genre_indicator` | 类型指标 | RPG/FPS/赛车等类型视觉标志 | 0.12% |
| 12 | `shows_platform` | 平台信息 | 主机/PC/手机/复古平台 | 2.75% |
| 13 | `has_text_element` | 文字元素 | 文本、Logo、品牌标识 | 0.45% |
| 14 | `features_creature` | 生物怪物 | 非人类生物、怪物、动物 | - |
| 15 | `has_additional_property` | 其他属性 | 未被上述覆盖的视觉特性 | - |

---

## 📝 详细说明

### 1. has_visual_style
**说明**: 游戏的整体艺术风格
**示例**:
- `cartoon` - 卡通风格
- `pixel_art` - 像素艺术
- `realistic_3d_graphics` - 写实3D画面
- `anime_style` - 日式动漫风格
- `cel_shaded` - 卡通渲染

**Phase 1变体**: has_visual_style, art_style, graphics_style

---

### 2. features_character
**说明**: 角色的视觉特征（最高频relation）
**示例**:
- `Sonic_the_Hedgehog` - 索尼克角色
- `armored_knight` - 身穿盔甲的骑士
- `space_marine_in_powered_armor` - 动力装甲太空战士
- `cartoon_character` - 卡通角色

**Phase 1变体**: features_character

---

### 3. set_in_environment
**说明**: 游戏场景和环境设定
**示例**:
- `fantasy_forest` - 奇幻森林
- `futuristic_space_station` - 未来空间站
- `medieval_castle` - 中世纪城堡
- `urban_cityscape` - 城市景观
- `underwater_scene` - 水下场景

**Phase 1变体**: set_in_environment

---

### 4. has_color_palette
**说明**: 主要色彩方案
**示例**:
- `bright_and_vibrant` - 明亮鲜艳
- `dark_blue_and_orange` - 深蓝与橙色
- `monochrome_black_and_white` - 黑白单色
- `neon_colors` - 霓虹色彩

**Phase 1变体**: has_color_palette, shows_color_palette

---

### 5. shows_perspective
**说明**: 游戏视角类型
**示例**:
- `first_person_view` - 第一人称
- `third_person_view` - 第三人称
- `top_down_view` - 俯视视角
- `side_scrolling` - 横版卷轴
- `isometric` - 等距视角

**Phase 1变体**: shows_perspective

---

### 6. has_ui_elements
**说明**: 用户界面可见元素
**示例**:
- `health_bar` - 血条
- `minimap` - 小地图
- `inventory_menu` - 物品栏
- `PlayStation_logo` - PlayStation标志
- `ESRB_rating` - ESRB评级

**Phase 1变体**: has_ui_elements, features_ui_element, includes_ui_elements等（19个变体）

---

### 7. has_atmosphere
**说明**: 游戏的情绪氛围
**示例**:
- `dark_and_gritty` - 黑暗严酷
- `whimsical_and_playful` - 奇幻有趣
- `tense_and_suspenseful` - 紧张悬疑
- `mystical_and_adventurous` - 神秘冒险

**Phase 1变体**: has_atmosphere, atmosphere

---

### 8. has_graphics_quality
**说明**: 图形质量和技术水平
**示例**:
- `high_definition` - 高清画质
- `low_poly` - 低多边形
- `realistic_lighting` - 写实光照
- `retro_graphics` - 复古画面

**Phase 1变体**: has_graphics_quality

---

### 9. features_vehicle
**说明**: 载具和交通工具
**示例**:
- `race_cars` - 赛车
- `spaceship` - 飞船
- `motorcycle` - 摩托车
- `medieval_horse` - 中世纪马匹

**Phase 1变体**: features_vehicle

---

### 10. features_weapon
**说明**: 武器和战斗装备
**示例**:
- `assault_rifle` - 突击步枪
- `medieval_sword` - 中世纪剑
- `laser_gun` - 激光枪
- `magic_staff` - 魔法杖

**Phase 1变体**: features_weapon, has_equipment（部分）

---

### 11. has_genre_indicator
**说明**: 游戏类型的视觉标志
**示例**:
- `racing_game_elements` - 赛车游戏元素
- `fighting_game_UI` - 格斗游戏界面
- `RPG_inventory_system` - RPG物品系统
- `first_person_shooter_crosshair` - FPS准星

**Phase 1变体**: game_type, features_game_type

---

### 12. shows_platform
**说明**: 游戏平台或时代
**示例**:
- `PlayStation_2` - PS2平台
- `Nintendo_64` - N64平台
- `PC` - PC平台
- `Xbox_One` - Xbox One
- `retro_8bit` - 复古8位

**Phase 1变体**: shows_platform, platform, has_platform, platform_era等

---

### 13. has_text_element
**说明**: 可见文字和品牌元素
**示例**:
- `game_title_on_cover` - 封面游戏标题
- `developer_logo` - 开发商Logo
- `warning_text` - 警告文字
- `subtitle_display` - 字幕显示

**Phase 1变体**: has_text, shows_text, has_logo, features_branding等

---

### 14. features_creature
**说明**: 非人类生物
**示例**:
- `dragon` - 龙
- `zombie` - 僵尸
- `alien_creature` - 外星生物
- `fantasy_monster` - 奇幻怪物
- `wildlife_animal` - 野生动物

**Phase 1变体**: features_creature（在Phase 1中未明确出现，但是重要类别）

---

### 15. has_additional_property
**说明**: 其他未分类的视觉属性
**示例**:
- `motion_blur_effect` - 运动模糊效果
- `particle_effects` - 粒子特效
- `customization_options` - 自定义选项
- 任何不属于上述14个类别的属性

**Phase 1变体**: 收集所有其他零散relations

---

## 🔄 与ML-1M的对比

| 维度 | ML-1M (电影) | Video Games (游戏) |
|------|--------------|-------------------|
| **核心数量** | 14 + 1 | 14 + 1 |
| **领域特点** | 电影情节、演员、场景 | 游戏玩法、界面、平台 |
| **独特relations** | - | `shows_platform`, `has_ui_elements`, `shows_perspective` |
| **共通relations** | 环境、氛围、角色、视觉风格、色彩 | 相同 |

---

## 📁 文件位置

- **Relation词表**: `results/videogames/relation_vocabulary_v1.json`
- **Phase 1结果**: `results/videogames/phase1_5percent_exploration.json`
- **分析脚本**: `scripts/analyze_videogames_relations.py`

---

## ✅ 下一步

使用这个relation词表进行Phase 2的**约束提取**：
```bash
python scripts/videogames_phase2_constrained_extraction.py \
    --vocabulary results/videogames/relation_vocabulary_v1.json \
    --percentage 100.0
```

这将使用14+1个标准relations对全部14,969个游戏进行知识提取。
