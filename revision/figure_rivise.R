###############################################################################
# mycoCirc — Visualization Script for Manuscript Figures
#
# Requirements: R >= 4.0, ggplot2, ggpubr, pheatmap, reshape2, scales, dplyr
# Usage: Rscript mycoCirc_figures.R
# Output: PDF figures suitable for publication
###############################################################################

workdir = "C://Data/_PanFungi/result/4_model/"
setwd(workdir)

library(ggplot2)
library(ggpubr)
library(reshape2)
library(scales)
library(dplyr)
library(tidyr)

# ---- Nature-style colour scheme (colourblind-safe) ----
col_modeA     <- "#0077BB"  # Nature blue   — Mode A / Genome+GTF
col_modeB     <- "#EE7733"  # Nature orange  — Mode B / +GeneExp
col_jedi      <- "#009988"  # Nature teal    — JEDI
col_circpcbl  <- "#CC3311"  # Nature red     — CircPCBL
col_single    <- "#0077BB"  # single-exon
col_multi     <- "#33BBEE"  # Nature cyan    — multi-exon
col_scratch   <- "#BBBBBB"  # Nature gray    — from scratch
col_pretrained<- "#0077BB"  # Nature blue    — pre-trained
col_candida   <- "#0077BB"
col_crypto    <- "#EE7733"
col_fila       <- "#009988"

# ---- Group name mapping (data file names -> display names) ----
GROUP_MAP <- c(
  "Candida"      = "Ascomycete yeast",
  "Cryptococcus" = "Basidiomycete yeast",
  "Filamentous"  = "Filamentous fungi"
)
remap_groups <- function(df, col = NULL) {
  if (is.null(col)) {
    col <- intersect(c("group", "Group"), colnames(df))
  }
  for (c in col) {
    df[[c]] <- ifelse(df[[c]] %in% names(GROUP_MAP),
                      GROUP_MAP[df[[c]]], df[[c]])
  }
  df
}

OUTPUT_DIR <- "manuscript/figures/"

# ===========================================================================
# Figure 1: Main Results — AUROC Comparison (Bar Plot)
# ===========================================================================
plot_main_results <- function() {
  df <- read.delim("comparison_model_benchmark.tsv", stringsAsFactors = FALSE)
  df <- remap_groups(df)

  methods_keep <- c("PanCirc (Genome+GTF)", "PanCirc (+GeneExp)", "JEDI", "CircPCBL")
  df <- df[df$method %in% methods_keep, ]
  df <- df[!grepl("HighDim|From-scratch", df$note), ]
  df$auroc <- as.numeric(df$auroc)
  df$group <- factor(df$group, levels = c("Ascomycete yeast", "Basidiomycete yeast", "Filamentous fungi"))

  df$method <- gsub("PanCirc \\(Genome\\+GTF\\)", "mycoCirc (Genome+GTF)", df$method)
  df$method <- gsub("PanCirc \\(\\+GeneExp\\)", "mycoCirc (+GeneExp)", df$method)
  df$method <- factor(df$method, levels = c("mycoCirc (Genome+GTF)", "mycoCirc (+GeneExp)", "JEDI", "CircPCBL"))

  p <- ggplot(df, aes(x = group, y = auroc, fill = method)) +
     geom_hline(yintercept = c(0.5, 0.7), color = "darkred", linetype = "dashed", linewidth = 0.5) +
    geom_col(position = position_dodge(0.55), width = 0.45) +
    geom_text(aes(label = sprintf("%.3f", auroc)),
              position = position_dodge(0.55), vjust = -0.3, size = 3.2) +
    scale_fill_manual(values = c("mycoCirc (Genome+GTF)" = col_modeA,
                                 "mycoCirc (+GeneExp)"  = col_modeB,
                                 "JEDI"                 = col_jedi,
                                 "CircPCBL"             = col_circpcbl)) +
    scale_y_continuous(limits = c(0, 0.90), expand = c(0, 0)) +
    labs(
         x = "Taxonomic Group", y = "AUROC",
         fill = "Method") +
    theme_bw(base_size = 12) +
    theme(legend.position = "right",
          plot.title = element_text(face = "bold"),
          panel.grid.major.x = element_blank())

  ggsave(file.path(OUTPUT_DIR, "Fig1_main_results.pdf"), p, width = 8, height = 2.8)
  message("Saved Fig1_main_results.pdf")
  return(p)
}

# ===========================================================================
# Figure 2: Component Ablation — AUROC Drop
# ===========================================================================
plot_component_ablation <- function() {
  df <- data.frame(
    Group = rep(c("Ascomycete yeast", "Basidiomycete yeast", "Filamentous fungi"), each = 6),
    Condition = rep(c("Full (Mode A)", "No GTF", "No GenomeCtx",
                      "No Species", "No Junction", "All zero"), 3),
    AUROC = c(0.7127, 0.6111, 0.7266, 0.7127, 0.7010, 0.5000,
              0.6927, 0.5511, 0.7124, 0.6927, 0.6807, 0.4945,
              0.6796, 0.6201, 0.6780, 0.6796, 0.6444, 0.5033)
  )

  df$Drop <- NA
  for (g in unique(df$Group)) {
    full <- df$AUROC[df$Group == g & df$Condition == "Full (Mode A)"]
    df$Drop[df$Group == g] <- df$AUROC[df$Group == g] - full
  }

  df$Condition <- factor(df$Condition,
                         levels = rev(c("Full (Mode A)", "No GTF", "No GenomeCtx",
                                       "No Species", "No Junction", "All zero")))

  p <- ggplot(df[df$Condition != "Full (Mode A)", ],
              aes(x = Group, y = Condition, fill = Drop)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%.4f", AUROC)), size = 2.8) +
    scale_fill_gradient2(low = col_jedi, mid = "white", high = col_modeA,
                         midpoint = 0, name = "AUROC\nChange") +
    labs(x = "", y = "") +
    theme_bw(base_size = 11) +
    theme(legend.position = "right",
          axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))

  ggsave(file.path(OUTPUT_DIR, "Fig2_component_ablation.pdf"), p, width = 3.8, height = 3.5)
  message("Saved Fig2_component_ablation.pdf")
  return(p)
}

# ===========================================================================
# Figure 3: Exon Type Breakdown — Single-exon vs Multi-exon AUROC
# ===========================================================================
plot_exon_breakdown <- function() {
  df <- read.delim("exon_type_auroc_summary.tsv", stringsAsFactors = FALSE)
  df <- remap_groups(df)
  df$auroc_single <- as.numeric(df$auroc_single)
  df$auroc_multi <- as.numeric(df$auroc_multi)
  df$group <- factor(df$group, levels = c("Ascomycete yeast", "Basidiomycete yeast", "Filamentous fungi"))

  x_labels <- sprintf("%s\n(SE: %.1f%%)", df$group, df$pct_single_exon_of_positive)

  df_plot <- df[, c("group", "auroc_single", "auroc_multi")]
  colnames(df_plot) <- c("Group", "Single-exon", "Multi-exon")
  df_plot <- melt(df_plot, id.vars = "Group", variable.name = "ExonType", value.name = "AUROC")

  p <- ggplot(df_plot, aes(x = Group, y = AUROC, fill = ExonType)) +
    geom_col(position = position_dodge(0.55), width = 0.35) +
    geom_text(aes(label = sprintf("%.3f", AUROC)),
              position = position_dodge(0.55), vjust = -0.3, size = 2.5) +
    scale_x_discrete(labels = setNames(x_labels, df$group)) +
    scale_fill_manual(values = c("Single-exon" = col_single,
                                 "Multi-exon"  = col_multi)) +
    scale_y_continuous(limits = c(0, 0.90), expand = c(0, 0)) +
    labs(x= "", y = "AUROC", fill = "Exon Type") +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))

  ggsave(file.path(OUTPUT_DIR, "Fig3_exon_breakdown.pdf"), p, width = 3.8, height = 3.2)
  message("Saved Fig3_exon_breakdown.pdf")
  return(p)
}

# ===========================================================================
# Figure 4: PM1 Cross-Species Generalisation
# ===========================================================================
plot_pm1_results <- function() {
  df <- read.delim("mycoCirc/results/pm1_cross_species_test.tsv", stringsAsFactors = FALSE)
  df <- remap_groups(df)

  df <- df[df$model_config == "default", ]
  df$mode <- gsub("Genome\\+GTF", "Mode A (Genome+GTF)", df$mode)
  df$mode <- gsub("\\+GeneExp", "Mode B (+GeneExp)", df$mode)
  df$mode <- factor(df$mode, levels = c("Mode A (Genome+GTF)", "Mode B (+GeneExp)"))

  df$group_label <- paste0(df$group, "~PM1")

  p <- ggplot(df, aes(x = group_label, y = auroc, fill = mode)) +
    geom_col(position = position_dodge(0.55), width = 0.35) +
    geom_text(aes(label = sprintf("%.3f", auroc)),
              position = position_dodge(0.55), vjust = -0.3, size = 3.2) +
    scale_fill_manual(values = c("Mode A (Genome+GTF)" = col_modeA,
                                 "Mode B (+GeneExp)"   = col_modeB)) +
    scale_y_continuous(limits = c(0, 0.85), expand = c(0, 0)) +
    labs(x = "Source Model", y = "AUROC",
         fill = "Inference\nMode") +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))

  ggsave(file.path(OUTPUT_DIR, "Fig4_pm1_results.pdf"), p, width = 4, height = 2.8)
  message("Saved Fig4_pm1_results.pdf")
  return(p)
}

# ===========================================================================
# Figure 5: Expression Ablation — CV AUROC
# ===========================================================================
plot_expression_ablation <- function() {
  df <- read.delim("comparison_ablation.tsv", stringsAsFactors = FALSE)
  df <- remap_groups(df)

  names(df)[names(df) == "cv_condition"] <- "condition"

  df$condition[df$condition == "full (CircExp+GeneExp)"] <- "Full (CircExp+GeneExp)"
  df$condition[df$condition == "shuffle CircExp"] <- "Shuffle CircExp"
  df$condition[df$condition == "shuffle GeneExp"] <- "Shuffle GeneExp"
  df$condition[df$condition == "zero both"] <- "Zero both"
  df$condition[df$condition == "no expression"] <- "No expression"

  df$condition <- factor(df$condition, levels = c("Full (CircExp+GeneExp)",
                                                   "Shuffle CircExp", "Shuffle GeneExp",
                                                   "Zero both", "No expression"))

  p <- ggplot(df, aes(x = condition, y = auroc, fill = group)) +
      geom_hline(yintercept = c(0.6,0.95), color = "darkred", linetype = "dashed", linewidth = 0.5) +
    geom_col(position = position_dodge(0.55), width = 0.5) +
    geom_text(aes(label = sprintf("%.3f", auroc)),
              position = position_dodge(0.55), vjust = -0.3, size = 2.1) +
    scale_fill_manual(values = c("Ascomycete yeast"      = col_candida,
                                 "Basidiomycete yeast"   = col_crypto,
                                 "Filamentous fungi"     = col_fila)) +
    scale_y_continuous(limits = c(0, 1.1), expand = c(0, 0)) +
    labs(         x = "Condition", y = "CV AUROC",
         fill = "Group") +
    theme_bw(base_size = 10) +
    theme(legend.position = "right",
          plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 25, hjust = 1))

  ggsave(file.path(OUTPUT_DIR, "Fig5_expression_ablation.pdf"), p, width = 8.5, height = 3)
  message("Saved Fig5_expression_ablation.pdf")
  return(p)
}

# ===========================================================================
# Figure 6: Feature Importance Heatmap
# ===========================================================================
plot_feature_importance <- function() {
  df <- read.delim("feature_importance_all.tsv", stringsAsFactors = FALSE)
  # Rename group columns (Candida/Cryptococcus/Filamentous -> display names)
  for (old_name in names(GROUP_MAP)) {
    if (old_name %in% colnames(df)) {
      colnames(df)[colnames(df) == old_name] <- GROUP_MAP[old_name]
    }
  }

  df <- df[!grepl("reserved_", df$feature), ]
  df <- df[df$feature != "biotype", ]

  df_melt <- melt(df, id.vars = "feature", variable.name = "Group", value.name = "Importance")
  df_melt$feature <- factor(df_melt$feature, levels = rev(unique(df_melt$feature)))

  p <- ggplot(df_melt, aes(x = Group, y = feature, fill = Importance)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%.2f", Importance)), size = 2.8) +
    scale_fill_gradient(low = "white", high = col_modeA, name = "IG\nAttribution") +
    labs(x = "", y = "Feature") +
    theme_bw(base_size = 10) +
    theme(legend.position = "right",
          axis.text.x = element_text(angle = 30, hjust = 0.5, vjust = 0.5))

  ggsave(file.path(OUTPUT_DIR, "Fig6_feature_importance.pdf"), p, width = 3.8, height = 2.8)
  message("Saved Fig6_feature_importance.pdf")
  return(p)
}

# ===========================================================================
# Figure 7: Junction Prediction Accuracy
# ===========================================================================
plot_junction_accuracy <- function() {
  df <- read.delim("junction_topk.tsv", stringsAsFactors = FALSE)
  df <- remap_groups(df)

  df_plot <- df[, c("group", "top1_fuzzy_5bp", "top3_fuzzy_5bp")]
  colnames(df_plot) <- c("Group", "Fuzzy Top-1", "Fuzzy Top-3")
  df_plot <- melt(df_plot, id.vars = "Group", variable.name = "Metric", value.name = "Accuracy")
  df_plot$Accuracy <- df_plot$Accuracy * 100

  p <- ggplot(df_plot, aes(x = Group, y = Accuracy, fill = Metric)) +
    geom_col(position = position_dodge(0.55), width = 0.35) +
    geom_text(aes(label = sprintf("%.1f%%", Accuracy)),
              position = position_dodge(0.55), vjust = -0.2, size = 2.5) +
    scale_fill_manual(values = c("Fuzzy Top-1" = col_single,
                                 "Fuzzy Top-3" = col_multi)) +
    scale_y_continuous(limits = c(0, 105), expand = c(0, 0)) +
    labs( x = "Group", y = "Accuracy (%)",
         fill = "Metric") +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))

  ggsave(file.path(OUTPUT_DIR, "Fig7_junction_accuracy.pdf"), p, width = 3.8, height = 3.2)
  message("Saved Fig7_junction_accuracy.pdf")
  return(p)
}

# ===========================================================================
# Figure 8: Pre-training Ablation — From-Scratch vs Pre-trained
# ===========================================================================
plot_pretrain_ablation <- function() {
  df <- read.delim("mycoCirc/results/comparison_fromscratch_full.tsv", stringsAsFactors = FALSE)
  df <- remap_groups(df)

  df_plot <- df[df$mode == "Mode_A", ]

  df_plot_long <- data.frame(
    Group = rep(df_plot$group, 2),
    Condition = c(rep("From Scratch", 3), rep("Pre-trained", 3)),
    AUROC = c(df_plot$from_scratch_auroc, df_plot$pretrained_auroc)
  )
  df_plot_long$Condition <- factor(df_plot_long$Condition, levels = c("From Scratch", "Pre-trained"))

  # Drop labels at precise positions above the taller bar per group
  drop_labels <- data.frame(
    Group = df_plot$group,
    y = pmax(df_plot$from_scratch_auroc, df_plot$pretrained_auroc) + 0.02,
    label = sprintf("Drop: %.4f", df_plot$auroc_diff)
  )

  p <- ggplot(df_plot_long, aes(x = Group, y = AUROC, fill = Condition)) +
    geom_col(position = position_dodge(0.55), width = 0.35) +
    geom_text(aes(label = sprintf("%.4f", AUROC)),
              position = position_dodge(0.55), vjust = -0.1, size = 3.2) +
    geom_text(data = drop_labels,
              aes(x = Group, y = y, label = label),
              vjust = 0, size = 3.2, col = col_jedi, inherit.aes = FALSE) +
    scale_fill_manual(values = c("From Scratch" = col_scratch,
                                 "Pre-trained"  = col_pretrained)) +
    scale_y_continuous(limits = c(0, 0.85), expand = c(0, 0)) +
    labs(         x = "", y = "AUROC", fill = "Condition") +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))

  ggsave(file.path(OUTPUT_DIR, "Fig8_pretrain_ablation.pdf"), p, width = 3.8, height = 3.5)
  message("Saved Fig8_pretrain_ablation.pdf")
  return(p)
}

# ===========================================================================
# Figure S1: Model Architecture Schematic
# ===========================================================================
plot_architecture_schematic_base <- function() {
  pdf(file.path(OUTPUT_DIR, "FigS1_architecture_schematic.pdf"),
      width = 10, height = 7)

  par(mar = c(0, 0, 0, 0))
  plot(0, 0, type = "n", xlim = c(0, 10), ylim = c(0, 10), axes = FALSE, xlab = "", ylab = "")

  draw_box <- function(x, y, w, h, text, col = "#E8E8E8", border = "#333333", cex = 0.8) {
    rect(x - w/2, y - h/2, x + w/2, y + h/2, col = col, border = border, lwd = 1.5)
    text(x, y, text, cex = cex, font = 2)
  }

  # Inputs (bottom)
  draw_box(1.5, 1.2, 1.8, 0.6, "Input: GTF\n(17 features)", col = "#B3CDE3")
  draw_box(4, 1.2, 1.8, 0.6, "Input: Genome Ctx\n(200x8 tensor)", col = "#B3CDE3")
  draw_box(6.5, 1.2, 1.8, 0.6, "Input: Junction\n(flank seqs)", col = "#B3CDE3")
  draw_box(8.8, 1.2, 1.4, 0.6, "Species ID", col = "#B3CDE3")

  # Encoders (middle)
  draw_box(1.5, 3.0, 1.8, 0.7, "GTFEncoder\n(10K params)", col = "#FDBF6F")
  draw_box(4, 3.0, 1.8, 0.7, "GenomicCtxEnc\n(156K params)", col = "#FDBF6F")
  draw_box(6.5, 3.0, 1.8, 0.7, "JunctionEncoder\n(246K params)", col = "#FDBF6F")
  draw_box(8.8, 3.0, 1.4, 0.7, "SpeciesEmb\n(3K params)", col = "#FDBF6F")

  arrows(1.5, 1.5, 1.5, 2.6, length = 0.1, lwd = 1.5)
  arrows(4, 1.5, 4, 2.6, length = 0.1, lwd = 1.5)
  arrows(6.5, 1.5, 6.5, 2.6, length = 0.1, lwd = 1.5)
  arrows(8.8, 1.5, 8.8, 2.6, length = 0.1, lwd = 1.5)

  text(1.5, 3.8, "128-dim", cex = 0.6, col = "gray40")
  text(4, 3.8, "128-dim", cex = 0.6, col = "gray40")
  text(6.5, 3.8, "64-dim", cex = 0.6, col = "gray40")
  text(8.8, 3.8, "32-dim", cex = 0.6, col = "gray40")

  arrows(1.5, 3.35, 1.5, 4.8, length = 0.1, lwd = 1.5)
  arrows(4, 3.35, 4, 4.8, length = 0.1, lwd = 1.5)
  arrows(6.5, 3.35, 6.5, 4.8, length = 0.1, lwd = 1.5)
  arrows(8.8, 3.35, 8.8, 4.8, length = 0.1, lwd = 1.5)

  draw_box(5.5, 6.0, 7, 1.2, "FusionModule (260K params)\nConcat -> MLP(352->256->128)", col = "#C4A5DE")

  lines(c(1.5, 1.5, 5.5), c(4.2, 5.4, 5.4), lwd = 1.5)
  lines(c(4, 4, 5.5), c(4.2, 5.4, 5.4), lwd = 1.5)
  lines(c(6.5, 6.5, 5.5), c(4.2, 5.4, 5.4), lwd = 1.5)
  lines(c(8.8, 8.8, 5.5), c(4.2, 5.4, 5.4), lwd = 1.5)

  arrows(5.5, 6.6, 5.5, 7.8, length = 0.1, lwd = 2)

  draw_box(5.5, 8.8, 3.5, 1.0, "GeneHead (8K params)\nLinear(128->1) + Sigmoid\n-> P(circRNA|gene)", col = "#FB9A99")

  text(5, 9.8, "mycoCirc Architecture (775,858 total parameters)", cex = 1.2, font = 2)

  dev.off()
  message("Saved FigS1_architecture_schematic.pdf")
}

# ===========================================================================
# Figure S2: K-mer Motif Enrichment Heatmap (sequence x group)
# ===========================================================================
plot_motif_enrichment <- function() {
  dir_names <- c("Candida", "Cryptococcus", "Filamentous")
  groups <- c("Ascomycete yeast", "Basidiomycete yeast", "Filamentous fungi")
  names(dir_names) <- groups
  df_list <- list()
  for (g in groups) {
    f <- paste0("mycoCirc/results/interpretability/", dir_names[g], "/motif_enrichment.tsv")
    d <- read.delim(f, stringsAsFactors = FALSE)
    d$group <- g
    df_list[[g]] <- d
  }
  df <- do.call(rbind, df_list)

  # Build matrix: motifs x groups
  all_motifs <- unique(df$motif)
  mat <- matrix(NA, nrow = length(all_motifs), ncol = length(groups),
                dimnames = list(all_motifs, groups))
  for (g in groups) {
    sub <- df[df$group == g, ]
    mat[sub$motif, g] <- sub$log2_enrichment
  }

  # Sort by mean enrichment across groups
  mat <- mat[order(rowMeans(mat, na.rm = TRUE), decreasing = TRUE), ]

  # Melt for ggplot
  hm <- reshape2::melt(mat, varnames = c("motif", "group"), value.name = "enrichment")
  hm$motif <- factor(hm$motif, levels = rev(rownames(mat)))
  hm$group <- factor(hm$group, levels = groups)

  p <- ggplot(hm, aes(x = group, y = motif, fill = enrichment)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = round(enrichment, 1)), size = 3, color = "gray30") +
    scale_fill_gradient2(low = "#0077BB", mid = "white", high = "#CC3311",
                         midpoint = 4.4, name = "log2 Enrichment") +
    labs(x = "", y = "3-mer Motif",
         title = "K-mer Motif Enrichment near Backsplice Junctions") +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold", size = 10),
          axis.text.x = element_text(angle = 0, hjust = 0.5),
          axis.text.y = element_text(family = "mono"))

  ggsave(file.path(OUTPUT_DIR, "FigS2_motif_enrichment.pdf"), p, width = 5, height = 5.5)
  message("Saved FigS2_motif_enrichment.pdf")
  return(p)
}
run_all <- function() {
  message("=== Generating mycoCirc manuscript figures ===\n")

  dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

  p1 <- plot_main_results()
  p2 <- plot_component_ablation()
  p3 <- plot_exon_breakdown()
  p4 <- plot_pm1_results()
  p5 <- plot_expression_ablation()
  p6 <- plot_feature_importance()
  p7 <- plot_junction_accuracy()
  p8 <- plot_pretrain_ablation()
  plot_architecture_schematic_base()
  p_s2 <- plot_motif_enrichment()

  message("\n=== All figures generated ===")
  message("Figures saved to: ", normalizePath(OUTPUT_DIR))
}

# Execute
run_all()

