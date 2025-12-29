# 知识提取使用指南

*Created: 2025-12-29*

---

## 📋 概述

本文档说明如何使用知识提取模块从电影海报中提取视觉知识点。

---

## 🚀 快速开始

### 1. 准备环境

```bash
# 安装依赖
pip install openai pillow tqdm

# 如果使用本地模型，还需要：
pip install transformers torch
```

### 2. 运行Pilot测试（10部电影）

```bash
python scripts/run_pilot_extraction.py \
    --poster_dir data/raw/ml-1m/posters \
    --id_mapping data/recbole/ml-1m/id_mappings.json \
    --backend openai \
    --model gpt-4o-mini \
    --num_samples 10 \
    --output results/pilot_extraction.json \
    --api_key YOUR_API_KEY
```

**输出**：
- `results/pilot_extraction.json` - 包含所有提取结果
- 控制台显示进度和统计信息

---

## 📚 模块说明

### PosterLoader - 海报加载器

**功能**：
- 使用ID映射加载海报（RecBole ID → 原始movie_id → 海报文件）
- 支持批量加载
- 支持随机采样
- 可选图片预处理（resize、格式转换）

**示例**：

```python
from src.extraction.poster_loader import PosterLoader

# 初始化
loader = PosterLoader(
    poster_dir='data/raw/ml-1m/posters',
    id_mapping_path='data/recbole/ml-1m/id_mappings.json'
)

# 随机采样20部电影
sample_ids = loader.sample_items(n=20, seed=42)

# 加载单张海报
poster = loader.load_poster(recbole_id=1, max_size=(1024, 1536))

# 批量加载
posters = loader.load_posters_batch(sample_ids, max_size=(1024, 1536))

# 获取item信息
info = loader.get_item_info(recbole_id=1)
# {'recbole_id': 1, 'original_movie_id': 1, 'poster_path': '...', 'poster_exists': True}
```

---

### MLLM Interface - 多模态LLM接口

**支持的后端**：
1. **OpenAI API** (推荐用于pilot)
   - GPT-4o (高质量，贵)
   - GPT-4o-mini (性价比高，推荐)

2. **本地模型** (开发中)
   - Qwen-VL
   - LLaVA

**示例 - OpenAI**：

```python
from src.extraction.mllm_interface import create_mllm
from PIL import Image

# 创建MLLM实例
mllm = create_mllm(
    backend='openai',
    model_name='gpt-4o-mini',
    api_key='YOUR_API_KEY'
)

# 提取知识
image = Image.open('poster.jpg')
output = mllm.extract_from_image(
    image=image,
    system_prompt="You are an expert...",
    user_prompt="Analyze this poster...",
    temperature=0.7,
    max_tokens=1000
)
```

**示例 - 本地模型** (占位，待实现):

```python
mllm = create_mllm(
    backend='local',
    model_name='qwen-vl',
    model_path='/path/to/model',
    device='cuda'
)
```

---

### Prompt Templates - 提示词模板

**Phase 1 - 自由探索**：

```python
from src.extraction.prompts import PromptTemplates

# 获取prompts
system_prompt = PromptTemplates.get_phase1_system_prompt()
user_prompt = PromptTemplates.get_phase1_user_prompt(movie_title="The Matrix")

# 解析输出
output = "color_scheme: dark_blue\nvisual_style: cyberpunk\n..."
knowledge_points = PromptTemplates.parse_extraction_output(output)
# [{'relation': 'color_scheme', 'entity': 'dark_blue'}, ...]

# 格式化输出
formatted = PromptTemplates.format_knowledge_points(knowledge_points, 'markdown')
print(formatted)
```

**Phase 3 - 受限提取** (使用词汇表):

```python
vocabulary = {
    'color_scheme': ['warm_tones', 'cool_tones', 'monochrome', ...],
    'visual_style': ['noir', 'minimalist', 'retro', ...]
}

system_prompt = PromptTemplates.get_phase3_system_prompt(vocabulary)
user_prompt = PromptTemplates.get_phase3_user_prompt(
    movie_title="The Matrix",
    relations=['color_scheme', 'visual_style']
)
```

---

## 🔧 Pilot测试脚本参数

### 必需参数

```bash
--poster_dir PATH       # 海报目录
--id_mapping PATH       # ID映射文件
--output PATH           # 输出JSON文件
```

### MLLM设置

```bash
--backend {openai,local}   # 后端类型 (default: openai)
--model MODEL_NAME         # 模型名称 (default: gpt-4o-mini)
--api_key KEY              # OpenAI API key (或设置OPENAI_API_KEY环境变量)
--temperature FLOAT        # 采样温度 (default: 0.7)
--max_tokens INT           # 最大token数 (default: 1000)
```

### 采样设置

```bash
--num_samples INT          # 采样数量 (default: 10)
--seed INT                 # 随机种子 (default: 42)
--sample_ids "1,2,3,..."   # 指定RecBole IDs (覆盖随机采样)
```

### 其他

```bash
--quiet                    # 静默模式
```

---

## 📊 输出格式

### JSON结构

```json
{
  "config": {
    "backend": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000,
    "num_samples": 10,
    "seed": 42,
    "timestamp": "2025-12-29T10:30:00"
  },
  "results": [
    {
      "recbole_id": 1,
      "original_movie_id": 1,
      "movie_title": "Toy Story (1995)",
      "num_knowledge_points": 12,
      "knowledge_points": [
        {"relation": "color_scheme", "entity": "bright_primary_colors"},
        {"relation": "visual_style", "entity": "animated_3d"},
        ...
      ],
      "raw_output": "color_scheme: bright_primary_colors\n...",
      "timestamp": "2025-12-29T10:30:15",
      "status": "success"
    },
    ...
  ]
}
```

---

## 💡 使用建议

### Pilot测试建议

1. **小规模测试** (10-20部电影)
   - 验证pipeline正常工作
   - 检查提取质量
   - 估算成本

2. **中等规模** (100部电影)
   - 评估知识点多样性
   - 为聚类准备数据
   - 成本约 $0.20-0.50

3. **Phase 1完整** (170部电影，5%)
   - 自由探索，构建初始词汇表
   - 成本约 $0.50-1.00

### 成本估算

**GPT-4o-mini定价** (截至2025年):
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

**每张海报估算**:
- Input: ~1500 tokens (image) + ~300 tokens (prompt) = ~1800 tokens
- Output: ~400 tokens (knowledge points)
- 成本: ~$0.0005/张 (0.5美分)

**批量成本**:
- 10部电影: ~$0.005 (0.5美分)
- 100部电影: ~$0.05 (5美分)
- 170部电影: ~$0.085 (8.5美分)
- 3416部电影: ~$1.70 (全量)

---

## 🐛 故障排除

### "FileNotFoundError: id_mappings.json"

确保已运行数据准备脚本：
```bash
python baselines/prepare_data_for_recbole.py \
    --ml_data_dir data/raw/ml-1m \
    --output_dir data/recbole/ml-1m \
    --min_interactions 5
```

### "OpenAI API key not found"

设置环境变量：
```bash
export OPENAI_API_KEY="your-api-key"
```

或使用 `--api_key` 参数。

### "Poster not found"

检查海报目录路径和文件名是否正确（应该是`{original_movie_id}.jpg`）。

---

## 📖 下一步

完成pilot测试后：
1. 分析提取结果质量
2. 调整prompt模板（如有必要）
3. 运行Phase 1完整提取（170部电影）
4. 实现知识点聚类（Phase 2）
5. 运行Phase 3受限提取（剩余电影）

---

*Last updated: 2025-12-29*
