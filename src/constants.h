/**
 * @file constants.h
 * @brief This file contains constants used across the CUDA rasterizer.
 * IMPORTANT: Use 'make clean' after changing a constant here.
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

/** Dataset file of polygons mapped to Hilbert space. */
#define MAPPED_CSV "T1NA_mapped.csv"
/** Output file of rasterized polygons. */
#define OUTPUT_CSV "rasterized.csv"
/** Output file of run results ready to plot. */
#define GRAPH_CSV "graph_results.csv"

/** The value of the cells outside the polygon. */
#define EMPTY_COLOR 0
/** The value of the cells on the border of the polygon. */
#define PARTIAL_COLOR 1
/** The value of the cells inside the polygon. */
#define FULL_COLOR 2
/** The value of the unknown cells (not rasterized yet). */
#define UNCERTAIN_COLOR 3
#define FULL_CHECKED 4

/** The size of the Hilbert space (2^16). */
#define HILBERT_SIZE 65536

#endif