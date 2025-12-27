================================================================================
项目文档说明
================================================================================

本目录包含项目的详细规划和实验设计文档。

由于markdown文件编码问题，我们使用Python文件来存储文档内容。
这些文件包含详细的中文文档字符串（docstring），可以直接运行查看。

================================================================================
文档列表
================================================================================

1. development_plan.py
   - 完整的开发计划
   - 包括所有baseline方法和创新方法的实现计划
   - API成本估算
   - 时间规划
   - 项目结构说明

   查看方式:
   ```bash
   python3 docs/development_plan.py
   ```

2. experiment_design.py
   - 详细的实验设计方案
   - 8种方法对比（4个baseline + 4个创新方法）
   - 消融实验设计
   - 开源模型（Qwen3-VL）验证方案
   - 评估指标和统计检验方法

   查看方式:
   ```bash
   python3 docs/experiment_design.py
   ```

================================================================================
快速导航
================================================================================

想了解:                    查看文件:
--------                   ---------
整体规划                   development_plan.py
实验设计                   experiment_design.py
Baseline方法               development_plan.py (方案A部分)
创新方法                   development_plan.py (方案B部分)
消融实验                   experiment_design.py (消融实验设计)
API成本                    development_plan.py (API成本估算)
评估指标                   experiment_design.py (评估指标)
Qwen3-VL方案              experiment_design.py (开源模型验证)

================================================================================
在代码中使用
================================================================================

这些文档文件也可以作为Python模块导入，获取配置常量:

```python
# 导入开发计划中的常量
from docs.development_plan import (
    KNOWLEDGE_CATEGORIES,
    BASELINE_METHODS,
    EVALUATION_METRICS,
    API_COST_ESTIMATES
)

# 导入实验配置
from docs.experiment_design import (
    EXPERIMENT_CONFIG,
    ABLATION_MATRIX,
    EXPECTED_PERFORMANCE
)

# 查看知识点类别
print(KNOWLEDGE_CATEGORIES)
# ['Color_Palette', 'Visual_Mood', ...]

# 查看实验方法
print(EXPERIMENT_CONFIG['methods'])
# ['LightGCN', 'KGAT', 'RippleNet', 'MKGAT', ...]
```

================================================================================
文档更新
================================================================================

最后更新: 2024-12-25
文档格式: Python文件（UTF-8编码）
维护者: 项目团队

如需更新文档，直接编辑对应的Python文件即可。
