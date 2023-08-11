# Run main_process_68k_pbmc.R from https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis first.
# The following writes out the count matrix and the class labels to be used by classifiers in this project.
writeMM(m_n_1000, 'count_matrix.in')
class_label <- lapply(cls_id, as.character)
lapply(class_label, write, "class_labels.in", append=TRUE, ncolumns=1000)