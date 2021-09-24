import numpy as np
from typing import List, Tuple, Optional, Union
from functools import reduce
import operator


def assign(
    alignments: List[List[int]],
    frequencies: List[float],
    travel_time: List[List[float]]
):
    def fu(f, u):
        if f == 0.0 and u == float('inf'):
            return 0.5

        return f * u

    n_nodes = len(travel_time)
    n_nodes_e = n_nodes * 2

    travel_time_e = np.zeros((n_nodes_e, n_nodes_e))
    travel_time_e[:n_nodes, :n_nodes] = travel_time

    idx_r = []
    idx_i = []
    idx_j = []

    for i, seq in enumerate(alignments):
        idx_r += [i] * (len(seq) - 1)
        idx_i += [node - 1 for node in seq[:-1]]
        idx_j += [node - 1 for node in seq[1:]]

    idx_io = [i + n_nodes for i in idx_i]
    idx_jo = [j + n_nodes for j in idx_j]

    idx_ix = idx_i + idx_io + idx_j
    idx_jx = idx_j + idx_i + idx_jo

    idx_r = reduce(operator.add, [idx_r] * 3)
    idx_i = idx_ix
    idx_j = idx_jx

    n_arcs = len(idx_r)

    arc_frequencies = [frequencies[r]
                       if idx_i[i] >= n_nodes else float('inf')
                       for i, r in enumerate(idx_r)]
    arc_costs = [travel_time_e[i][j] for i, j in zip(idx_i, idx_j)]

    u = [[float('inf') for _ in range(n_nodes_e)]
         for _ in range(n_nodes_e)]

    for dst in range(n_nodes, n_nodes * 2):
        u[dst][dst] = 0
        f = [0 for _ in range(n_nodes_e)]

        idx_r_cp = idx_r.copy()
        idx_i_cp = idx_i.copy()
        idx_j_cp = idx_j.copy()

        arc_frequencies_cp = arc_frequencies.copy()
        arc_costs_cp = arc_costs.copy()

        for arc in range(n_arcs):
            costs = [u[j][dst] + arc_cost for j, arc_cost in zip(idx_j_cp, arc_costs_cp)]
            cost = min(costs)
            idx = costs.index(cost)

            i = idx_i_cp[idx]
            fa = arc_frequencies_cp[idx]

            if u[i][dst] >= cost:
                if fa == float('inf'):
                    u[i][dst] = cost
                    f[i] = fa
                else:
                    u[i][dst] = (fu(f[i], u[i][dst]) + fa * cost) / (f[i] + fa)
                    f[i] = f[i] + fa

            idx_r_cp.pop(idx)
            idx_i_cp.pop(idx)
            idx_j_cp.pop(idx)
            arc_frequencies_cp.pop(idx)
            arc_costs_cp.pop(idx)

    u = [[val if val != float('inf') else 0 for val in row[n_nodes:]] for row in u[n_nodes:]]
    return u


def total_trip_time(
    alignments: List[List[int]],
    allocations: List[int],
    travel_time: np.ndarray,
    complete_trip: bool = False
) -> Tuple[List[List[float]], List[List[float]]]:

    # unidirectional = (np.tril(travel_time) > 0).sum() == 0
    # n_routes = len(alignments)
    # n_nodes = (len(travel_time[0]) + 1) // 2

    trip_times = []

    for alignment in alignments:
        trip_times.append(0.0)

        prev_node = None
        for node in alignment:
            if prev_node is None:
                prev_node = node
                continue

            # TODO: Fix this
            # if node > n_nodes - 1 and prev_node < n_nodes:
            #     trip_times[-1] += travel_time[prev_node - 1][2 * n_nodes - 1 - prev_node]
            #     prev_node = 2 * n_nodes - prev_node

            trip_times[-1] += travel_time[prev_node - 1][node - 1]
            prev_node = node

        if complete_trip and alignment[-1] != alignment[0]:
            from_, to_ = alignment[-1], alignment[0]

            trip_times[-1] += travel_time[from_ - 1][to_ - 1]

    frequencies = [1 / (trip_time / allocation + 1e-20)
                   for trip_time, allocation
                   in zip(trip_times, allocations)]

    return trip_times, frequencies


def augment_alignment(alignments, n_nodes):
    new_alignments = []

    for alignment in alignments:
        new_alignments.append([])

        # determine where does the alignment start becoming reverse direction
        node_prev = None
        for i, node in enumerate(alignment):
            if node_prev is not None and node - node_prev < 0:
                if i == 1:
                    new_alignments[-1][-1] = 2 * n_nodes - node_prev
                elif node_prev != n_nodes:
                    new_alignments[-1].append(2 * n_nodes - node_prev)

                for node in alignment[i:]:
                    new_alignments[-1].append(2 * n_nodes - node)
                
                break
            else:
                new_alignments[-1].append(node)

            node_prev = node

    return new_alignments


def augment_travel_time(mat):
    n_nodes = len(mat)
    n_nodes_e = n_nodes * 2 - 1

    new_mat = []
    for i in range(n_nodes_e):
        new_mat.append([])

        for j in range(n_nodes_e):
            if j > i:
                if j < n_nodes:
                    new_mat[-1].append(mat[i][j])
                elif j == n_nodes_e - i - 1:
                    new_mat[-1].append(0)
                elif i >= n_nodes - 1:
                    new_mat[-1].append(mat[n_nodes_e - i - 1][n_nodes_e - j - 1])
                else:
                    new_mat[-1].append(float('inf'))
            else:
                new_mat[-1].append(0)

    return new_mat


def reward(alignments,
           allocations,
           travel_time: Union[np.ndarray, List[List[float]]],
           demands: Optional[Union[np.ndarray, List[List[float]]]] = None,
           log_costs=None):

    def bi2uni(mat):
        n_nodes = (len(mat) + 1) // 2

        out = []
        for i in range(n_nodes):
            out.append([])

            for j in range(n_nodes):
                if j > i:
                    out[i].append(mat[i][j])
                elif j < i:
                    out[i].append(mat[2 * n_nodes - 2 - i][2 * n_nodes - 2 - j])
                else:
                    out[i].append(0)

        return out

    n_nodes = len(travel_time)

    if type(travel_time) is np.ndarray:
        travel_time = travel_time.tolist()

    log_costs_new = log_costs if log_costs is not None else {}

    unidirectional = sum(sum(travel_time[i][j] for j in range(n_nodes) if j < i)
                         for i in range(n_nodes)) == 0

    if not unidirectional:
        travel_time = augment_travel_time(travel_time)
        alignments = augment_alignment(alignments, n_nodes)

    try:
        log_costs_new[None]
    except:
        full_alignments = alignments[:1]
        full_allocations = [sum(allocation for allocation in allocations)]

        trip_times, frequencies = total_trip_time(full_alignments, full_allocations, travel_time)
        us = assign(full_alignments, frequencies, travel_time)
        us = bi2uni(us) if not unidirectional else us
        log_costs_new[None] = us

    if len(alignments) == 1:
        return log_costs_new[None], log_costs_new

    assert len(alignments) == 2, "Only one express service is supported"
    assert len(alignments[0]) == n_nodes * 2 - 1, "First route as to serve all stops"
    assert len(allocations) == 2, "Only one express service is supported"

    log_idx = (tuple(alignments[1]), allocations[1])

    try:
        log_costs_new[log_idx]
    except:
        trip_times, frequencies = total_trip_time(alignments, allocations, travel_time)
        us = assign(alignments, frequencies, travel_time)
        us = bi2uni(us) if not unidirectional else us
        log_costs_new[log_idx] = us

    us = log_costs_new[log_idx]
    usor = log_costs_new[None]

    if demands is not None:
        return (
            sum(us[i][j]
                for i in range(len(us))
                for j in range(len(us[0]))
                if us[i][j] != 0) /
            sum(usor[i][j]
                for i in range(len(us))
                for j in range(len(us[0]))),
            us,
            log_costs_new
        )

    return us, log_costs_new
