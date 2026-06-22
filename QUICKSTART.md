# mycoCirc 用户使用教程

本教程说明如何用 mycoCirc 对一个**新真菌基因组**进行 circRNA 基因预测。

---

## 1 安装

```bash
# 克隆仓库
git clone git@github.com:yukkikou/mycoCirc.git
cd mycoCirc

# 创建环境（推荐 micromamba）
micromamba create -f scripts/env_micromamba.yml
micromamba activate mycoCirc

# 或者用 pip
pip install -r requirements.txt
```

## 2 下载预训练权重

权重文件已包含在仓库中，位于 `model_weights/` 目录：

| 组 | 最适合的真菌 | 文件 |
|:---|:-------------|:-----|
| Candida | _Candida_, _Pichia_, _Saccharomyces_ | `model_weights/mycoCirc_candida.pt` |
| Cryptococcus | _Cryptococcus_ | `model_weights/mycoCirc_cryptococcus.pt` |
| Filamentous | _Fusarium_, _Aspergillus_, _Neurospora_, _Penicillium_ 等丝状真菌 | `model_weights/mycoCirc_filamentous.pt` |

```bash
# 克隆仓库即包含权重
git clone git@github.com:yukkikou/mycoCirc.git
cd mycoCirc
ls model_weights/  # 直接可用
```

## 3 准备输入文件

mycoCirc 只需要两个文件：
- **基因组 FASTA**（`.fa` / `.fasta`）
- **基因注释 GTF**（`.gtf` / `.gff3`）

> 可从 Ensembl Fungi (https://fungi.ensembl.org) 或 NCBI 下载。

## 4 运行预测

```bash
python scripts/predict.py \
    --genome my_genome.fa \
    --gtf my_annotation.gtf \
    --checkpoint mycoCirc_filamentous.pt \
    --config config/default.yaml \
    --output predictions.tsv
```

### 参数说明

| 参数 | 必需 | 说明 |
|:-----|:----:|:------|
| `--genome` | ✅ | 参考基因组 FASTA |
| `--gtf` | ✅ | 基因注释 GTF |
| `--checkpoint` | ✅ | 预训练权重文件 |
| `--config` | ✅ | 模型配置文件（用 `config/default.yaml`） |
| `--output` | ✅ | 输出 TSV 路径 |
| `--genexp` | ❌ | 基因表达 CSV（可选） |
| `--device` | ❌ | `cuda` 或 `cpu`（默认自动检测） |
| `--batch-size` | ❌ | 批大小（默认 32） |

## 5 输出结果解读

输出的 TSV 包含每列：

```
gene_id      chrom     start      end        strand  p_circ    n_exons
B9J08_000001 scaffold1  1000       3000       +       0.2345    1
B9J08_000002 scaffold1  5000       8000       -       0.8765    4
```

- **p_circ**: 该基因产生 circRNA 的概率（0–1），≥0.5 为预测阳性
- **n_exons**: 外显子数（单外显子=1，多外显子≥2）

预测结束后会打印统计摘要：
```
Probability distribution: min=0.0012, max=0.9876, mean=0.3124, median=0.1876
Genes predicted to produce circRNA (p>=0.5): 321/4500
```

## 6 常见问题

**Q: 对不同真菌该选哪个权重？**
A: 参考分组对应表。如果不确定，Candida 权重（紧凑基因组）和 Filamentous 权重（复杂基因组）是较保守的选择。

**Q: 结果中有很多 p_circ ≈ 0.5 的基因，怎么办？**
A: 可以调整阈值。建议用 0.5 作为默认阈值，低置信度结果用 `0.6` 或 `0.7` 过滤。

**Q: 我的真菌不属于这三个组（如毛霉门）？**
A: 建议用 Filamentous 权重（真菌多样性最高）或 Candida 权重（基础 Ascomycota 特征）。

**Q: 需要 GPU 吗？**
A: 不需要，CPU 也可以运行（速度约 100 基因/秒）。GPU 约 500 基因/秒。
