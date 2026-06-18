#!/bin/bash
# ==========================================
# mycoCirc — GitHub Upload Script
# ==========================================
# 使用方法：
#   bash scripts/upload_github.sh
#
# 前提：已配置 SSH key 到 GitHub 账户
#   ssh -T git@github.com  # 确认显示认证成功
# ==========================================

set -e

GITHUB_USER="yukkikou"
REPO_NAME="mycoCirc"
GIT_EMAIL="xueyanhu@pku.edu.cn"
GIT_NAME="yukkikou"

cd /media/share/workdir/1610305236/Panfungi/4_model

echo "=========================================="
echo "  mycoCirc → GitHub 上传脚本"
echo "=========================================="

# ── Step 0: 检查 SSH 认证 ──────────────────
echo ""
echo ">>> Step 0: 检查 SSH 认证"
if ssh -T git@github.com 2>&1 | grep -q "successfully"; then
    echo "  ✅ SSH 认证正常"
else
    echo "  ❌ SSH 认证失败，请先配置 SSH key"
    exit 1
fi

# ── Step 1: 初始化 Git ──────────────────────
echo ""
echo ">>> Step 1: 初始化 Git 仓库"
if [ -d ".git" ]; then
    echo "  .git 已存在，跳过 init"
else
    git init
fi

git config user.email "$GIT_EMAIL"
git config user.name "$GIT_NAME"
echo "  Git 用户: $GIT_NAME"

# ── Step 2: 确保 .gitignore 覆盖临时文件 ──
echo ""
echo ">>> Step 2: 整理 .gitignore"
grep -qxF "checkpoints/" .gitignore 2>/dev/null || echo "checkpoints/" >> .gitignore
grep -qxF "logs/" .gitignore 2>/dev/null || echo "logs/" >> .gitignore
grep -qxF "other_models/" .gitignore 2>/dev/null || echo "other_models/" >> .gitignore
echo "  ✅ .gitignore 已更新"

# ── Step 3: 清理旧暂存 ──────────────────────
echo ""
echo ">>> Step 3: 添加文件"
git rm -r --cached . 2>/dev/null || true

git add .gitignore LICENSE README.md requirements.txt
git add config/ data/ model/ train/ utils/ interpret/ scripts/ docs/

# 添加关键结果文件（小文件，GitHub 上可查看）
git add results/comparison_model_benchmark.tsv \
        results/ablations.tsv \
        results/comparison_ablation.tsv \
        results/comparison_fromscratch_full.tsv \
        results/significance_tests.tsv \
        results/hyperparameter_sweep_summary.tsv \
        results/pm1_cross_species_test.tsv \
        results/exon_type_auroc.tsv \
        results/exon_type_auroc_summary.tsv \
        results/interpretability/junction_topk.tsv \
        results/interpretability/feature_importance_all.tsv || true

# 添加图
git add figures/fig_comparison.pdf \
        figures/fig_modality_ablation.pdf || true

# ── Step 4: 首次提交 ────────────────────────
echo ""
echo ">>> Step 4: 提交"
if git diff --cached --quiet; then
    echo "  无变更需要提交"
else
    git commit -m "Initial commit: mycoCirc — Pan-fungal circRNA foundation model

Multi-modal architecture integrating gene annotation, genomic context,
and junction-level sequence features for circRNA gene prediction.
Pre-trained on 22 fungal strains (Candida, Cryptococcus, Filamentous),
with group-specific fine-tuning and 5-fold cross-validation.

Key results:
- AUROC 0.70 on held-out test species
- Cross-species generalisation to Talaromyces marneffei AUROC 0.6955
- GTFEncoder identified as dominant modality (ablation drop -0.10)
- RNA-seq-free inference from genome + GTF only"
    echo "  ✅ 提交成功"
fi

# ── Step 5: 创建 GitHub 仓库 ────────────────
echo ""
echo ">>> Step 5: 创建 GitHub 仓库"
if git remote get-url origin 2>/dev/null; then
    echo "  远程仓库已存在"
else
    git remote add origin "git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
    echo "  ✅ 远程仓库已添加 (SSH)"
fi

# ── Step 6: 推送 ─────────────────────────────
echo ""
echo ">>> Step 6: 推送到 GitHub"
echo ""
echo "=========================================="
echo "  本地仓库准备就緒！"
echo ""
echo "  运行以下命令推送："
echo ""
echo "    git push -u origin main"
echo ""
echo "  如果 GitHub 默认分支是 master："
echo "    git branch -M main"
echo "    git push -u origin main"
echo "=========================================="
