// #include "rwalk.h"
#include <omp.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <math.h>

unsigned long int random_seed()
{
    struct timeval tv;
    gettimeofday(&tv,0);
    return (tv.tv_sec + tv.tv_usec);
}

bool isin_neighborhood(int node, int const* neighs, int ptr_i, int ptr_j) {
    for (int i = ptr_i; i < ptr_j; ++i) {
        if (neighs[i] == node) {
            return true;
        }
    }
    return false;
}

void biased_random_walk(int const* ptr, int const* neighs, int n, int num_walks, int num_steps, double p, double q, int seed, int nthreads, int* walks) 
{
    const gsl_rng_type *G;
    gsl_rng **r;
    G = gsl_rng_default;

    double const max_prob = fmax(1.0 / p, fmax(1.0, 1.0 / q));
    double const prob_0 = 1.0 / p / max_prob;
    double const prob_1 = 1.0 / max_prob;
    double const prob_2 = 1.0 / q / max_prob;

    if (nthreads > 0) {
        omp_set_num_threads(nthreads);
    }
    else{
        nthreads = omp_get_max_threads();
        omp_set_num_threads(nthreads);
    }
    
    if (seed < 0){
        seed = random_seed();
    }

    r = (gsl_rng **) malloc(nthreads * sizeof(gsl_rng *));

    for (int thread_i = 0; thread_i < nthreads; thread_i++){
        r[thread_i] = gsl_rng_alloc(G);
        gsl_rng_set(r[thread_i],seed*thread_i);
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++) 
    {
        int offset, num_neighs, r_step, thread_j;
        thread_j = omp_get_thread_num();
        for (int walk = 0; walk < num_walks; walk++) 
        {
            int curr_pos = i;
            int prev_pos = i;
            offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
            walks[offset] = i;

            for (int step = 0; step < num_steps; step++)
            {
                num_neighs = ptr[curr_pos + 1] - ptr[curr_pos];
                if (num_neighs > 0)
                {
                    int new_pos;
                    double r_r;
                    int ptr_i = ptr[prev_pos];
                    int ptr_j = ptr[prev_pos+1];

                    while (1)
                    {
                        if (num_neighs == 1){
                            new_pos = neighs[ptr[curr_pos]];
                            break;
                        }

                        r_r = gsl_rng_uniform(r[thread_j]);
                        r_step = gsl_rng_uniform_int(r[thread_j], num_neighs);
                        new_pos = neighs[ptr[curr_pos] + r_step];

                        if ((step == 0) || (p == 1.0 && q == 1.0)) {
                            break;
                        }
                        if (new_pos == prev_pos) {
                            if (r_r < prob_0) {//bias towards returning to the previous node
                                break;
                            }
                        }
                        else if (isin_neighborhood(new_pos, neighs, ptr_i, ptr_j)){
                            if (r_r < prob_1) {//bias towards moving to the neighborhood of the previous node
                                break;
                            }
                        }
                        else if (r_r < prob_2) {//bias towards moving outside the neighborhood of the previous node
                            break;
                        }
                    }
                    prev_pos = curr_pos;
                    curr_pos = new_pos;
                }
                walks[offset + step + 1] = curr_pos;
            }
        }
    }
#pragma omp barrier

    for (int thread_i = 0; thread_i < nthreads; thread_i++)
    {
        r[thread_i] = gsl_rng_alloc(G);
        gsl_rng_free(r[thread_i]);
    }
    free(r);
}