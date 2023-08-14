# Run main_process_68k_pbmc.R from https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis first.
# The following writes out the count matrix, the class labels to be used by classifiers in this project, as well as the gene names.
writeMM(m_n_1000, 'count_matrix.in')

class_label <- lapply(cls_id, as.character)
lapply(class_label, write, "class_labels.in", append=TRUE, ncolumns=1000)

genes_ordered_by_var <- lapply(use_genes_n_id, as.character)
lapply(genes_ordered_by_var, write, "ordered_gene_names.in", append=TRUE, ncolumns=1000)