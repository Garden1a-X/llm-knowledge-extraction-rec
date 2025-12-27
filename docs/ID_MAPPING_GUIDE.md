# ID映射使用指南

*Created: 2025-12-27*

---

## 📋 背景

MovieLens 1M的原始movie_id是不连续的（如：1, 2, 3, 5, 10, 11, 17, ...），经过5-core过滤后会更加稀疏。

为了避免RecBole内部重新映射ID导致海报对应关系错误，我们在数据准备阶段就手动将ID映射为连续整数。

---

## 📁 映射文件位置

```
data/recbole/ml-1m/id_mappings.json
```

运行数据准备脚本时自动生成：
```bash
python baselines/prepare_data_for_recbole.py \
    --ml_data_dir data/raw/ml-1m \
    --output_dir data/recbole/ml-1m \
    --min_interactions 5
```

---

## 📊 映射文件结构

```json
{
  "user_id_map": {
    "original_to_new": {"1": 1, "2": 2, "3": 3, ...},
    "new_to_original": {"1": 1, "2": 2, "3": 3, ...}
  },
  "item_id_map": {
    "original_to_new": {"1": 1, "2": 2, "5": 3, "10": 4, ...},
    "new_to_original": {"1": 1, "2": 2, "3": 5, "4": 10, ...}
  },
  "stats": {
    "num_users": 6040,
    "num_items": 3706,
    "num_ratings": 1000209,
    "min_interactions": 5
  }
}
```

**关键字段**：
- `original_to_new`: 原始ID → RecBole连续ID
- `new_to_original`: RecBole连续ID → 原始ID（⭐ 加载海报时用这个）

---

## 🎯 使用场景

### 场景1: 加载电影海报

```python
import json
from PIL import Image

# 1. 加载映射文件
with open('data/recbole/ml-1m/id_mappings.json', 'r') as f:
    mappings = json.load(f)

item_id_map = mappings['item_id_map']['new_to_original']

# 2. RecBole给你的item_id（连续的）
recbole_item_id = 123  # 比如推荐结果中的item

# 3. 映射回原始movie_id
original_movie_id = item_id_map[str(recbole_item_id)]
# str(recbole_item_id) 因为JSON key必须是字符串

# 4. 加载对应的海报
poster_path = f'data/raw/ml-1m/posters/{original_movie_id}.jpg'
poster = Image.open(poster_path)
```

### 场景2: 批量提取电影知识点

```python
import json
import pandas as pd

# 加载映射
with open('data/recbole/ml-1m/id_mappings.json', 'r') as f:
    mappings = json.load(f)

item_id_map = mappings['item_id_map']['new_to_original']

# RecBole的item列表（连续ID: 1, 2, 3, 4, ...）
recbole_item_ids = range(1, mappings['stats']['num_items'] + 1)

# 转换为原始movie_id用于提取知识点
extraction_tasks = []
for recbole_id in recbole_item_ids:
    original_id = item_id_map[str(recbole_id)]
    extraction_tasks.append({
        'recbole_id': recbole_id,
        'original_movie_id': original_id,
        'poster_path': f'data/raw/ml-1m/posters/{original_id}.jpg'
    })

df = pd.DataFrame(extraction_tasks)
print(f"Total items to extract: {len(df)}")
```

### 场景3: 反向查找（原始ID → RecBole ID）

```python
# 如果你知道原始movie_id，想知道RecBole中的ID
original_to_new = mappings['item_id_map']['original_to_new']

original_movie_id = 1  # Toy Story (1995)
recbole_id = original_to_new[str(original_movie_id)]
print(f"Movie {original_movie_id} -> RecBole ID {recbole_id}")
```

---

## ⚠️ 注意事项

### 1. **JSON键都是字符串**
```python
# ❌ 错误
original_id = item_id_map[123]  # KeyError!

# ✅ 正确
original_id = item_id_map[str(123)]
```

### 2. **RecBole ID从1开始，不是0**
```python
# RecBole使用的item_id范围: [1, num_items]
# 0 是保留的padding ID

for recbole_id in range(1, num_items + 1):  # ✅ 从1开始
    original_id = item_id_map[str(recbole_id)]
```

### 3. **5-core过滤会减少item数量**
```python
# 原始ML-1M: 3883 movies
# 5-core过滤后: ~3700 movies（实际数字看过滤结果）

# 不是所有原始movie_id都在映射中！
# 只有过滤后保留的movie才会有映射
```

---

## 🔍 验证映射正确性

```python
import json

with open('data/recbole/ml-1m/id_mappings.json', 'r') as f:
    mappings = json.load(f)

item_map = mappings['item_id_map']

# 检查映射是否一致
print("验证映射一致性:")
for new_id_str, orig_id in item_map['new_to_original'].items():
    assert item_map['original_to_new'][str(orig_id)] == int(new_id_str)

print("✅ 映射一致性检查通过！")

# 打印统计信息
print(f"\nStats:")
print(f"  Users: {mappings['stats']['num_users']}")
print(f"  Items: {mappings['stats']['num_items']}")
print(f"  Ratings: {mappings['stats']['num_ratings']}")
print(f"  Min interactions: {mappings['stats']['min_interactions']}-core")

# 查看几个样例
print(f"\n样例映射（RecBole ID -> Original Movie ID）:")
for i in range(1, min(11, mappings['stats']['num_items'] + 1)):
    orig = item_map['new_to_original'][str(i)]
    print(f"  {i} -> {orig}")
```

---

## 📚 相关文件

- `baselines/prepare_data_for_recbole.py` - 生成映射的脚本
- `data/recbole/ml-1m/id_mappings.json` - 映射文件
- `data/recbole/ml-1m/ml-1m.item` - RecBole item文件（已使用连续ID）
- `data/raw/ml-1m/posters/*.jpg` - 海报文件（文件名是原始movie_id）

---

*Last updated: 2025-12-27*
