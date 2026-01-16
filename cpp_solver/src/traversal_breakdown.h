#ifndef TRAVERSAL_BREAKDOWN_H
#define TRAVERSAL_BREAKDOWN_H

#include <map>
#include <set>
#include <tuple>
#include <vector>
#include <deque>
#include <ostream>

#include <armadillo>

#include "utils.h"

void build_workers_pool(
    Instance& inst,
    const std::vector<std::tuple<float, int>>& candidate_tasks,
    const std::set<int>& available_machines,
    const std::set<int>& available_operators,
    std::vector<std::tuple<int, int>>& workers_pool,
    std::map<std::tuple<int, int>, int>& worker2poolindex_map
);

void build_workers_conflict_matrix(
    const std::vector<std::tuple<int, int>>& workers_pool,
    arma::SpMat<float>& W_conflicts
);

void build_task_worker_compatibility_matrix(
    Instance& inst,
    const std::vector<std::tuple<float, int>>& candidate_tasks,
    std::map<std::tuple<int, int>, int>& worker2poolindex_map,
    arma::Mat<float>& T_W_compat,
    std::map<int, int>& task2poolindex_map
);

arma::Col<float> build_task_scores_vector(
    const std::vector<std::tuple<float, int>>& candidate_tasks
);

int count_worker_conflicts(
    const arma::SpMat<float>& W_conflicts,
    const arma::Col<float>& active_workers
);

int compute_and_log_conflict_graph_connectivity(
    const arma::SpMat<float>& W_conflicts,
    int nb_workers,
    std::ostream& log_stream
);

int prune_conflicting_workers(
    const arma::SpMat<float>& W_conflicts,
    arma::Mat<float>& T_W_compat,
    const arma::Col<float>& T_scores,
    arma::Col<float>& active_workers,
    std::ostream& log_stream
);

std::vector<int> workers_sorted_by_versatility(
    const arma::Mat<float>& T_W_compat,
    const arma::Col<float>& active_workers
);

int greedy_assign_tasks_to_workers(
    Instance& inst,
    Solution& sol,
    std::map<int, std::deque<int>>& job_stacks,
    const std::vector<std::tuple<float, int>>& candidate_tasks,
    const std::vector<std::tuple<int, int>>& workers_pool,
    const std::map<int, int>& task2poolindex_map,
    const std::vector<int>& worker_choice_order,
    arma::Mat<float>& T_W_compat,
    arma::Col<float>& active_workers,
    int time_pos,
    std::set<int>& available_machines,
    std::set<int>& available_operators,
    std::map<int, int>& next_time_persue_job,
    std::map<int, std::set<int>>& release_calendar_machines,
    std::map<int, std::set<int>>& release_calendar_operators,
    std::map<int, int>& cumulative_remaining_time_per_job,
    std::ostream& log_stream
);

#endif
