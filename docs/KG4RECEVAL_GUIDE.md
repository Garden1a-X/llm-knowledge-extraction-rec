# KG4RecEval 使用指南

## 📖 简介

[KG4RecEval](https://github.com/HotBento/KG4RecEval) 是一个系统性评估知识图谱对推荐系统贡献的框架。

**论文**: "KG4RecEval: Does Knowledge Graph Really Matter for Recommender Systems?" (TOIS 2025)
- arXiv: https://arxiv.org/abs/2404.03164
- GitHub: https://github.com/HotBento/KG4RecEval

---

## 🎯 为什么使用KG4RecEval？

### 优势

1. **统一的KG推荐baseline**
   - 支持8个主流KG推荐模型：KGAT, KGCN, RippleNet, CFKG, CKE, KGNNLS, KTUP, KGIN
   - 基于RecBole 1.1.1，配置统一
   - 避免手动实现和调试

2. **解决我们的KGAT问题**
   - 我们之前KGAT失败（DGL版本不兼容）
   - KG4RecEval已经集成好了KGAT
   - 只需要替换数据集

3. **公平对比**
   - 统一的评估协议
   - 相同的数据分割
   - 标准化的指标计算

### KG4RecEval的关键发现

⚠️ **重要发现**：论文发现很多KG推荐模型其实并没有真正有效利用KG！
- KGAT的KGER（KG utilization efficiency）= **-0.026**（负值表示KG反而有害）
- 移除KG、随机扰动KG，甚至对cold-start用户，很多模型性能并不下降

**这对我们意味着什么**：
- ✅ 如果我们的方法能真正利用好KG，就是重要创新！
- ✅ 可学习Mask机制可能是关键（过滤无用的KG信息）
- ✅ 论文中可以引用这个发现，强调我们的方法优势

---

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/HotBento/KG4RecEval.git
cd KG4RecEval

# 安装依赖
conda create -n kg4rec python=3.9
conda activate kg4rec

pip install torch==2.0.0
pip install recbole==1.1.1
pip install dgl==0.9.1
pip install -r requirements.txt

# 修复numpy兼容性问题
# 在 recbole/evaluator/metrics.py 中：
# 将 np.float 替换为 float
```

**依赖版本**：
- Python 3.9
- PyTorch 2.0.0
- RecBole 1.1.1
- DGL 0.9.1

---

### 2. 数据准备

KG4RecEval使用RecBole格式，**这正好是我们已有的格式**！

#### 我们需要的文件

```
data/recbole/ml-1m/
├── ml-1m.inter          # ✅ 已有（User-Item交互）
├── ml-1m.item.kg        # ✅ 已有（Item知识图谱）
└── ml-1m.link           # ✅ 已有（Item-Entity映射）
```

**注意**：KG4RecEval不需要`ml-1m.user.kg`（用户兴趣图谱），只需要item侧的KG。

#### 文件格式验证

```bash
# 检查文件格式
head -5 data/recbole/ml-1m/ml-1m.inter
# 预期格式：user_id:token  item_id:token  rating:float  timestamp:float

head -5 data/recbole/ml-1m/ml-1m.item.kg
# 预期格式：head_id:token  relation_id:token  tail_id:token

head -5 data/recbole/ml-1m/ml-1m.link
# 预期格式：item_id:token  entity_id:token
```

---

### 3. 配置文件

复制我们的数据到KG4RecEval目录：

```bash
# 在KG4RecEval目录下
mkdir -p dataset/ml-1m
cp /path/to/your/data/recbole/ml-1m/*.* dataset/ml-1m/
```

创建配置文件 `config/ml-1m_kgat.yaml`：

```yaml
# 基础配置
dataset: ml-1m
data_path: dataset/

# KG配置
load_col:
  inter: [user_id, item_id, rating, timestamp]
  kg: [head_id, relation_id, tail_id]
  link: [item_id, entity_id]

# 数据分割（对齐我们的设置）
eval_args:
  split: {'RS': [0.7, 0.1, 0.2]}  # Per-user Random Split
  order: TO  # Time-based ordering
  group_by: user
  mode: full

# 评估指标
metrics: ['Recall', 'NDCG', 'Hit', 'Precision']
topk: [5, 10, 20]
valid_metric: NDCG@10

# KGAT模型配置
embedding_size: 64
kg_embedding_size: 64
layers: [64, 32, 16]
mess_dropout: 0.1
reg_weight: 0.0001

# 训练配置
epochs: 300
train_batch_size: 2048
learner: adam
learning_rate: 0.001
stopping_step: 10
```

---

### 4. 运行KGAT

```bash
# 运行KGAT
python main.py \
  --model=KGAT \
  --dataset=ml-1m \
  --config_files=config/ml-1m_kgat.yaml

# 查看结果
# 结果会保存在 log/KGAT/ 目录下
```

---

## 📊 运行其他KG模型

KG4RecEval支持8个模型，我们可以一键运行：

```bash
# KGCN
python main.py --model=KGCN --dataset=ml-1m \
  --config_files=config/ml-1m_kgcn.yaml

# RippleNet
python main.py --model=RippleNet --dataset=ml-1m \
  --config_files=config/ml-1m_ripplenet.yaml

# CKE
python main.py --model=CKE --dataset=ml-1m \
  --config_files=config/ml-1m_cke.yaml

# CFKG
python main.py --model=CFKG --dataset=ml-1m \
  --config_files=config/ml-1m_cfkg.yaml
```

**推荐运行的模型**（按重要性）：
1. **KGAT** - 最相关，图注意力网络
2. **KGCN** - 知识图谱卷积网络
3. **CKE** - 协同知识嵌入
4. **RippleNet** - 传播网络

---

## 🎯 明天的执行计划

### Plan: 使用KG4RecEval跑KGAT baseline

#### Step 1: 环境准备（30分钟）
```bash
# 1. 克隆KG4RecEval
cd ~/projects
git clone https://github.com/HotBento/KG4RecEval.git
cd KG4RecEval

# 2. 安装依赖
conda create -n kg4rec python=3.9 -y
conda activate kg4rec
pip install torch==2.0.0
pip install recbole==1.1.1
pip install dgl==0.9.1
pip install pandas numpy pyyaml tqdm

# 3. 修复numpy兼容性
# 找到recbole安装路径
python -c "import recbole; print(recbole.__file__)"
# 编辑 recbole/evaluator/metrics.py
# 将所有 np.float 替换为 float
```

#### Step 2: 数据准备（10分钟）
```bash
# 复制数据
mkdir -p dataset/ml-1m
cp ~/llm-knowledge-extraction-rec/data/recbole/ml-1m/ml-1m.inter dataset/ml-1m/
cp ~/llm-knowledge-extraction-rec/data/recbole/ml-1m/ml-1m.item.kg dataset/ml-1m/ml-1m.kg
cp ~/llm-knowledge-extraction-rec/data/recbole/ml-1m/ml-1m.link dataset/ml-1m/

# 验证文件
ls -lh dataset/ml-1m/
head -3 dataset/ml-1m/ml-1m.inter
head -3 dataset/ml-1m/ml-1m.kg
head -3 dataset/ml-1m/ml-1m.link
```

#### Step 3: 创建配置（10分钟）
创建 `config/ml-1m_kgat.yaml`（见上面的配置示例）

#### Step 4: 运行KGAT（20-30分钟）
```bash
python main.py --model=KGAT --dataset=ml-1m \
  --config_files=config/ml-1m_kgat.yaml

# 后台运行（推荐）
nohup python main.py --model=KGAT --dataset=ml-1m \
  --config_files=config/ml-1m_kgat.yaml \
  > kgat_ml1m.log 2>&1 &

# 监控进度
tail -f kgat_ml1m.log
```

#### Step 5: 提取结果（5分钟）
```bash
# 查看结果日志
cat log/KGAT/ml-1m/*.log | grep "test result"

# 提取NDCG@10
grep "NDCG@10" log/KGAT/ml-1m/*.log
```

---

## ⚠️ 可能的问题与解决

### 问题1: DGL版本不兼容

**症状**：运行KGAT报错 `AttributeError: 'DGLGraph' object has no attribute...`

**解决**：
```bash
# 确保DGL版本正确
pip install dgl==0.9.1 --force-reinstall
```

### 问题2: RecBole版本不兼容

**症状**：`np.float` 报错

**解决**：
```bash
# 找到recbole安装路径
python -c "import recbole; print(recbole.__file__)"

# 编辑 recbole/evaluator/metrics.py
# 全局替换: np.float → float
sed -i 's/np\.float/float/g' /path/to/recbole/evaluator/metrics.py
```

### 问题3: 数据加载失败

**症状**：`No columns has been loaded from [kg]`

**解决**：
1. 检查文件名是否正确（`ml-1m.kg` 而不是 `ml-1m.item.kg`）
2. 检查header格式（需要`:token`后缀）
3. 检查config中的`load_col`配置

### 问题4: 内存不足

**症状**：`CUDA out of memory`

**解决**：
```yaml
# 减小batch size
train_batch_size: 1024  # 从2048减到1024
```

---

## 📈 预期结果对比

基于KG4RecEval论文的发现，KGAT在ML-1M上的典型结果：

| 模型 | NDCG@10 | 说明 |
|------|---------|------|
| BPR (RecBole) | 0.1219 | 我们已有 |
| LightGCN (RecBole) | 0.1267 | 我们已有 |
| **KGAT (KG4RecEval)** | ~0.12-0.13 | 待运行 |
| **Ours-Full** | **0.1549** | 我们的方法 |

**如果KGAT结果在0.12-0.13**：
- ✅ 我们的方法（0.1549）**显著超越** KGAT
- ✅ 证明LLM提取的KG比传统KG更有效
- ✅ 可学习Mask机制有效过滤噪声

**论文写作要点**：
- 引用KG4RecEval的发现：传统KG推荐模型常常不能有效利用KG
- 对比KGAT的KGER=-0.026（负值），强调我们的方法真正利用了KG
- 归功于：LLM提取的高质量KG + Mask机制过滤噪声

---

## 🔄 与我们方法的对比

| 维度 | KGAT | Ours-Full |
|------|------|-----------|
| **KG来源** | 外部KG（Freebase/DBpedia） | LLM提取（GPT-4o-mini） |
| **KG质量** | 通用，噪声多 | 领域特定，高质量 |
| **噪声处理** | 无 | 可学习Mask |
| **用户兴趣** | 无 | User-Entity图谱 |
| **多视图** | 单一KG视图 | CF + KG双视图对比 |
| **预期性能** | 0.12-0.13 | 0.1549 ✅ |

---

## 💡 总结

### 为什么使用KG4RecEval？

1. ✅ **快速获得KGAT baseline**（避免手动实现的痛苦）
2. ✅ **统一评估协议**（公平对比）
3. ✅ **强化论文贡献**（引用KG4RecEval的发现）

### 明天的优先级

**建议顺序**：
1. **方法改进**（负采样、温度、对比学习）- 2-3小时
2. **KG4RecEval跑KGAT** - 1-1.5小时
3. **消融实验** - 0.5小时
4. **结果分析** - 1-2小时

**理由**：
- 方法改进可能提升5-10%，收益最大
- KGAT baseline增强论文说服力
- 消融实验是论文核心，必须做

---

## 📚 参考资料

- **KG4RecEval论文**: https://arxiv.org/abs/2404.03164
- **KG4RecEval GitHub**: https://github.com/HotBento/KG4RecEval
- **RecBole文档**: https://recbole.io/
- **KGAT原论文**: Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation" (KDD 2019)

---

**创建时间**: 2026-01-13 凌晨
**状态**: Ready for execution
**预计完成时间**: 1-1.5小时
