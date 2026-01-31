# Amazon Beauty 数据集准备指南

## 概述

- **数据集**: Amazon Beauty (All_Beauty)
- **特点**: 不做5-core过滤，使用全部交互（稀疏数据集，适合验证冷启动性能）
- **数据路径**: `/data/xuao/llm-knowledge-extraction-rec/data/raw/amazon-beauty`

---

## Step 1: 下载原始数据

从 Amazon Reviews 2023 下载数据：
- 网址: https://amazon-reviews-2023.github.io/

需要下载的文件：
```
All_Beauty.jsonl.gz        # 评论/交互数据
meta_All_Beauty.jsonl.gz   # 商品元数据
```

放置到：
```
/data/xuao/llm-knowledge-extraction-rec/data/raw/amazon-beauty/
```

---

## Step 2: 数据预处理

将原始数据转换为 RecBole 格式：

```bash
python scripts/prepare_beauty_dataset.py
```

**输出目录**: `/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-beauty/`

**生成文件**:
- `amazon-beauty.inter` - 用户-商品交互
- `amazon-beauty.item` - 商品信息
- `amazon-beauty.user` - 用户信息
- `id_mappings.json` - ID映射表
- `filtered_metadata.json` - 过滤后的元数据（用于后续提取）

---

## Step 3: 下载商品图片

```bash
python scripts/download_beauty_images.py
```

**输出目录**: `/data/xuao/llm-knowledge-extraction-rec/data/recbole/amazon-beauty/images/`

---

## Step 4: 提取视觉知识（LLM）

参照 Video Games 的提取流程，需要：

1. **Phase 1**: 探索性提取（5%样本）- 确定关系类型
2. **Phase 2**: 词汇标准化 - 建立标准词汇表
3. **Phase 3**: 词汇验证 - 验证覆盖率
4. **Phase 4**: 全量提取 - 使用标准词汇表提取所有商品
5. **Phase 5**: 用户兴趣提取

*待创建对应脚本*

---

## Step 5: 构建知识图谱

*待创建脚本*

---

## Step 6: 运行实验

### Baseline 方法
- BPR
- LightGCN
- KGAT
- VBPR
- MKGAT
- MMGCN

### Ours 方法
- Ours-Full

---

## 数据集特点

| 特性 | Beauty | ML-1M | Video Games |
|------|--------|-------|-------------|
| 5-core | ❌ 不过滤 | ✅ 过滤 | ✅ 过滤 |
| 密度 | 稀疏 | 中等 | 中等 |
| 冷启动 | 多 | 少 | 少 |
| 优势 | 验证KG对冷启动的帮助 | - | - |

---

## 文件清单

### 已创建
- [x] `scripts/prepare_beauty_dataset.py` - 数据预处理
- [x] `scripts/download_beauty_images.py` - 图片下载

### 待创建
- [ ] `scripts/beauty_phase1_extraction.py` - 探索性提取
- [ ] `scripts/beauty_phase4_extraction.py` - 全量提取
- [ ] `scripts/build_beauty_kg.py` - 构建知识图谱
- [ ] `scripts/extract_beauty_user_interests.py` - 用户兴趣提取
- [ ] `configs/ours_full_beauty.yaml` - 训练配置

---

*Created: 2026-01-29*
