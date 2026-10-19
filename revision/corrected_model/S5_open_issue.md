# 待办：S5 (Schizosaccharomyces pombe) 的归属（2026-10-17 记录，暂不处理）

## 事实
- 注册表 `all_lib_model_full.tsv` 中 S5 存在，`excluded=False`，但**既不在 TRAIN 也不在 TEST**。
- 数据完整：基因组、GTF、CircInfo（341 raw → 308 过滤 → **265 阳性基因**）、CircExp、GeneExp 全部存在；特征缓存已提取（`checkpoints/features/S5/`）。
- 预训练日志（17 株）与微调划分均不含 S5。
- git 历史、README、日志中**没有任何说明排除理由**——属划分定义时的遗漏。

## 影响
- 稿件 [23] 段称 "S. pombe is **strategically integrated**" 入 Filamentous 组，与代码不符。
- 稿件称 Filamentous 组含 8 个物种（A1,A2,A3,F3,F4,F6,N1,S5）：实际 5 训练 + 1 测试(F6) + 1 排除(A2) + 1 未用(S5)。

## 待定方案（作者 2026-10-17：先不做）
- A. 纳入训练：加入预训练（18 株）+ Filamentous 微调（6 株）→ 全重训 + 全部评估 + atlas 重做（约 1 天）。正文表述即可成立。
- B. 作为额外 held-out 测试物种：无需重训，只评估一遍；正文改为"额外 held-out 物种"。
- C. 仅加入微调（不重跑预训练）：成本中等，但预训练/微调面板不一致，需文中说明。

## 注意
若选 A 或 C，当前 `revision/corrected_model/` 下所有结果（含 atlas）需重跑。
