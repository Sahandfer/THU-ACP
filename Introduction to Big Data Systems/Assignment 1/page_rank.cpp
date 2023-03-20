#include "page_rank.h"
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "./common/CycleTimer.h"
#include "./common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

    /* Implement the page rank algorithm here.  You
    are expected to parallelize the algorithm using openMP.  Your
    solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode:

    // initialization: see example code below
    score_old[vi] = 1/numNodes;

    while (!converged and iter < MAXITER) {

      // compute score_new[vi] for all nodes vi:
      score_new[vi] = sum over all nodes vj reachable from incoming edges
              { score_old[vj] / number of edges leaving vj  }
      score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

      score_new[vi] += sum over all nodes vj with no outgoing edges
              { damping * score_old[vj] / numNodes }

      // compute how much per-node scores have changed
      // quit once algorithm has converged
      global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
      converged = (global_diff < convergence)
    }
    */

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    double *solution_new = new double[numNodes];
    double *score_old = solution;
    double *score_new = solution_new;
    bool converged = false;
    double broadcastScore = 0.0;
    double globalDiff = 0.0;
    int iter = 0;

// First optimization
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numNodes; ++i)
    {
        solution[i] = equal_prob;
    }

    while (!converged && iter < MAXITER)
    {
        iter++;
        broadcastScore = 0.0;
        globalDiff = 0.0;

// Second optimization
#pragma omp parallel for reduction(+ \
                                   : broadcastScore) schedule(guided, 512)
        for (int i = 0; i < numNodes; ++i)
        {
            score_new[i] = 0.0;

            if (outgoing_size(g, i) == 0)
            {
                broadcastScore += score_old[i];
            }
            const Vertex *in_begin = incoming_begin(g, i);
            const Vertex *in_end = incoming_end(g, i);
            for (const Vertex *v = in_begin; v < in_end; ++v)
            {
                score_new[i] += score_old[*v] / outgoing_size(g, *v);
            }
            score_new[i] = damping * score_new[i] + (1.0 - damping) * equal_prob;
        }

// Third optimization
#pragma omp parallel for reduction(+ \
                                   : globalDiff) schedule(static)
        for (int i = 0; i < numNodes; ++i)
        {
            score_new[i] += damping * broadcastScore * equal_prob;
            globalDiff += std::abs(score_new[i] - score_old[i]);
        }
        converged = (globalDiff < convergence);
        std::swap(score_new, score_old);
    }
    if (score_new != solution)
    {
        memcpy(solution, score_new, sizeof(double) * numNodes);
    }
    delete[] solution_new;
}
