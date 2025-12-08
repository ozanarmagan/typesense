#include <complex>
#include <limits>
#include <immintrin.h>
#include "vamana.h"
#include "glog/logging.h"

vamana_node_t::vamana_node_t(size_t R, size_t dims) {
    neighbors.reserve(R);
    vector.reserve(dims);
}

vamana_node_t::vamana_node_t(size_t R, const std::vector<float>& vector) : vector(vector) {
    neighbors.reserve(R);
}

Vamana::Vamana(const size_t R, const distance_metric_t metric, size_t dims) : R(R), metric(metric),
                                                                              dims(dims), medoid_tracker(dims, 0) {

}

const std::unordered_map<uint32_t, vamana_node_t>& Vamana::get_node_map() const {
    return node_map;
}

bool hasDuplicates(const std::vector<uint32_t>& neighbors) {
    std::unordered_set<uint32_t> seen;
    for (const auto& n: neighbors) {
        if (!seen.insert(n).second) {
            return true;  // Found a duplicate
        }
    }
    return false;
}

void Vamana::greedy_search(const uint32_t start, const std::vector<float>& query, const size_t k,
                           const size_t L, search_result_t& search_result,
                           FilterFn filter) {

    auto start_node_it = node_map.find(start);
    if (start_node_it == node_map.end()) {
        return;
    }

    // -------- thread‑local, zero‑alloc priority–queues ----------------------
    // if (filter && !(*filter)(start)) {
    //     return;
    // }

    using candidate_pq_t = std::priority_queue<
            vamana_candidate_t,
            std::vector<vamana_candidate_t>,
            vamana_candidate_t::min_cmp>;

    using result_pq_t = std::priority_queue<
            vamana_candidate_t,
            std::vector<vamana_candidate_t>,
            vamana_candidate_t::max_cmp>;

    /* One instance per thread → no allocations or locks on the hot path.      *
     * We simply “reset” them each call with an empty brace‑init, which is an  *
     * O(1) operation that keeps the comparator object intact.                 */
    static thread_local candidate_pq_t candidates;
    static thread_local result_pq_t results;

    candidates = {};   // cheap reset
    results = {};   // cheap reset

    VisitedSet* visited = g_visited_pool.acquire(node_map.size());

    candidates.emplace(start, distance_functions_t::compute(metric, start_node_it->second.vector, query, dims));
    visited->mark(start);

    float max_distance = std::numeric_limits<float>::max();

    while (!candidates.empty()) {
        auto nn = candidates.top();

        if (nn.distance > max_distance) {
            // we have atleast L results and the top remaining candidate is worse than the worst result so far
            break;
        }

        candidates.pop();

        // if (filter && !(*filter)(nn.id)) {
        //     continue;
        // }

        {
            auto it = node_map.find(nn.id);
            if (it != node_map.end()) {                     // pointer is now valid
                const auto* pNbr = it->second.neighbors.data();
                _mm_prefetch(reinterpret_cast<const char*>(pNbr), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(pNbr) + 64,_MM_HINT_T0);
            }
        }

        // Accept `nn` if we still need more points **or** it is closer than the current worst‐case candidate.
        if ((results.size() < L || nn.distance < results.top().distance) &&
            delete_list.find(nn.id) == delete_list.end()) {
            if (!filter || (*filter)(nn.id)) {
                results.emplace(nn.id, nn.distance);
            }
            // Drop the worst
            if (results.size() > L) {
                results.pop();
            }

            // When the queue has L elements, the element at `top()` is the worst,
            // so its distance is our new pruning radius.
            if (results.size() == L) {
                max_distance = results.top().distance;
            }
        }

        auto nodes_it = node_map.find(nn.id);
        if (nodes_it == node_map.end()) {
            continue;
        }

        const auto& ids = nodes_it->second.neighbors;
        constexpr int PREFETCH_DISTANCE = 4;   // tune: 2–8 is typical

        for (size_t i = 0; i < ids.size(); ++i) {

            /* ----------- prefetch the *vector* of the neighbour that we
             *           will reach PREFETCH_DISTANCE iterations later    */
            if (i + PREFETCH_DISTANCE < ids.size()) {
                uint32_t preId = ids[i + PREFETCH_DISTANCE];
                auto pit = node_map.find(preId);          // pointer known *now*
                if (pit != node_map.end()) {
                    _mm_prefetch(reinterpret_cast<const char*>
                                 (pit->second.vector.data()), _MM_HINT_T0);
                }
            }
            /* ---------- 1. test visited bitmap (tiny, hot) ----------- */
            if (!visited->mark(ids[i])) continue;

            /* ---------- 2. actually touch the prefetched vector ------ */
            auto nit = node_map.find(ids[i]);
            if (nit == node_map.end()) continue;

            float d = distance_functions_t::compute(metric,
                                                    nit->second.vector,
                                                    query, dims);
            candidates.emplace(ids[i], d);
        }
    }

    while (!results.empty()) {
        search_result.nearest_nodes.push_back(results.top());
        results.pop();
    }

    std::reverse(search_result.nearest_nodes.begin(), search_result.nearest_nodes.end());

    // we only need the top k results
    search_result.nearest_nodes.resize(std::min(k, search_result.nearest_nodes.size()));

    g_visited_pool.release(visited);   // give it back to the pool
}

void Vamana::robust_prune(uint32_t p, std::vector<vamana_candidate_t>& candidates, float max_alpha) {
    auto nodes_it = node_map.find(p);
    if (nodes_it == node_map.end()) {
        return;
    }

    auto& pneighbors = nodes_it->second.neighbors;
    pneighbors.clear();

    for (size_t loop_i = 0; loop_i < 2; loop_i++) {
        float alpha = (loop_i == 0) ? 1.0f : max_alpha;

        // candidates are already sorted in distance closest to `p` first
        for (size_t i = 0; i < candidates.size(); i++) {
            // Skip if we've already reached desired number of neighbors
            if (pneighbors.size() >= R) {
                return;
            }

            // Skip if this candidate has been pruned
            if (candidates[i].distance == std::numeric_limits<float>::lowest()) {
                continue;
            }

            // Skip self-loops
            if (candidates[i].id == p) {
                continue;
            }

            // Add current closest point to neighbors
            pneighbors.push_back(candidates[i].id);
            candidates[i].distance = std::numeric_limits<float>::lowest();

            // Get the vector for the newly added neighbor
            auto neighbor_it = node_map.find(candidates[i].id);
            if (neighbor_it == node_map.end()) {
                continue;
            }
            const auto& neighbor_vec = neighbor_it->second.vector;

            // Prune remaining candidates based on α-RNG property
            for (size_t j = i + 1; j < candidates.size(); j++) {
                if (candidates[j].distance == std::numeric_limits<float>::lowest()) {  // skip already pruned
                    continue;
                }

                if (candidates[j].id == p) {  // skip self
                    continue;
                }

                auto candidate_it = node_map.find(candidates[j].id);
                if (candidate_it == node_map.end()) {
                    continue;
                }
                const auto& candidate_vec = candidate_it->second.vector;

                float dist_between_candidates = distance_functions_t::compute(metric, neighbor_vec,
                                                                              candidate_vec, dims);

                // Apply α-RNG property: if the distance through the new neighbor
                // multiplied by alpha is less than the direct distance, prune the candidate
                if (alpha * dist_between_candidates <= candidates[j].distance) {
                    candidates[j].distance = std::numeric_limits<float>::lowest();  // mark as pruned
                }
            }
        }
    }
}

void Vamana::update_neighbors(uint32_t id, const std::vector<float>& vec, float alpha) {
    const auto& neighbors = node_map.at(id).neighbors;

    // Update all neighbors to potentially include this node in their neighbor lists
    for (uint32_t neighbor_id: neighbors) {
        if (delete_list.find(neighbor_id) != delete_list.end()) {
            continue;
        }

        auto nit = node_map.find(neighbor_id);
        if (nit == node_map.end()) continue;             // node already gone

        auto& neighbor_node = nit->second;

        if (neighbor_node.neighbors.size() >= R) {
            // If neighbor already has max connections, need to run pruning
            std::vector<vamana_candidate_t> ncandidates;
            ncandidates.reserve(neighbor_node.neighbors.size() + 1);

            for (auto& n_neighbor: neighbor_node.neighbors) {
                if (delete_list.find(n_neighbor) != delete_list.end()) {
                    continue;
                }

                auto n_it = node_map.find(n_neighbor);
                if (n_it == node_map.end()) {
                    continue;
                }

                float dist = distance_functions_t::compute(metric, n_it->second.vector,
                                                           neighbor_node.vector, dims);
                ncandidates.emplace_back(n_neighbor, dist);
            }

            float dist_to_node = distance_functions_t::compute(metric, neighbor_node.vector, vec, dims);
            ncandidates.emplace_back(id, dist_to_node);
            std::sort(ncandidates.begin(), ncandidates.end(), vamana_candidate_t::max_cmp());
            robust_prune(neighbor_id, ncandidates, alpha);
        } else {
            /* ------------------------------------------------------- *
             * 2.  Fast‑path when there is still room                  *
             *     → append only if not already present                *
             * ------------------------------------------------------- */
            if (std::find(neighbor_node.neighbors.begin(),
                          neighbor_node.neighbors.end(),
                          id) == neighbor_node.neighbors.end()) {
                neighbor_node.neighbors.push_back(id);
            }
        }
    }
}

void Vamana::insert(uint32_t id, const std::vector<float>& vec, size_t L, float alpha) {
    // Add the new point p to the graph
    node_map.try_emplace(id, R, vec);

    try_medoid_compute(vec);

    // Call greedy_search to get visited nodes
    search_result_t search_result;
    greedy_search(start_node, vec, L, L, search_result, nullptr);

    // Assign out neighbors of p after pruning
    robust_prune(id, search_result.nearest_nodes, alpha);

    // Update neighbor relationships
    update_neighbors(id, vec, alpha);
}

void Vamana::update(uint32_t id, const std::vector<float>& new_vec, size_t L, float alpha) {
    auto it = node_map.find(id);
    if (it == node_map.end() || delete_list.find(id) != delete_list.end()) {
        // Can't update a non-existing or deleted node
        return;
    }

    // 1. Overwrite vector
    it->second.vector = new_vec;

    // 2. Perform greedy search to find candidate neighbors
    search_result_t search_result;
    greedy_search(start_node, new_vec, L, L, search_result, nullptr);

    // 3. Robust prune to set out-neighbors of 'id'
    robust_prune(id, search_result.nearest_nodes, alpha);

    // 4. Update neighbor relationships
    update_neighbors(id, new_vec, alpha);
}

void Vamana::batch_delete() {
    if (delete_list.empty()) return;

    for (auto& [id, node]: node_map) {
        auto& nbrs = node.neighbors;
        nbrs.erase(std::remove_if(nbrs.begin(), nbrs.end(),
                                  [&](uint32_t x) { return delete_list.count(x); }),
                   nbrs.end());
    }

    delete_list.clear();
}


void Vamana::remove(uint32_t id) {
    /*------------------------------------------------------------------
     | IP‑DiskANN  Algorithm 5 :  in‑place deletion                    |
     *----------------------------------------------------------------*/

    auto it = node_map.find(id);
    if (it == node_map.end()) {
        return;                 // already gone
    }

    constexpr size_t L_del = 128;   // beam‑width for delete search
    constexpr size_t K_del = 50;    // candidates kept
    constexpr uint32_t C = 3;      // edges copied per anchor

    /*--------------------------------------------------------------*/
    /* 1. Local search around p                                     */
    /*--------------------------------------------------------------*/
    search_result_t sr;
    greedy_search(start_node, it->second.vector, K_del, L_del, sr);

    /*--------------------------------------------------------------*/
    /* 2. Helpers                                                   */
    /*--------------------------------------------------------------*/
    // scratch buffer lives once for the whole function
    std::vector<std::pair<float, uint32_t>> buf;      // (dist, id)

    auto select_top_c = [&](uint32_t anchor,
                            std::vector<uint32_t>& out) {
        out.clear();
        const auto& a_vec = node_map.at(anchor).vector;
        buf.clear();
        buf.reserve(sr.nearest_nodes.size());

        for (const auto& cand: sr.nearest_nodes) {
            if (cand.id == id) continue; // skip p
            if (!node_map.count(cand.id)) continue; // deleted
            float d = distance_functions_t::compute(
                    metric, a_vec,
                    node_map.at(cand.id).vector, dims);
            buf.emplace_back(d, cand.id);
        }
        size_t take = std::min<size_t>(C, buf.size());
        std::partial_sort(buf.begin(), buf.begin() + take,
                          buf.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        out.reserve(take);
        for (size_t i = 0; i < take; ++i) out.push_back(buf[i].second);
    };

    auto patch_edges = [&](uint32_t owner,
                           const std::vector<uint32_t>& add) {
        auto& nbrs = node_map.at(owner).neighbors;
        nbrs.insert(nbrs.end(), add.begin(), add.end());
        dedup_vector(nbrs);

        if (nbrs.size() > R) {  // re‑prune (rare)
            std::vector<vamana_candidate_t> cand;
            cand.reserve(nbrs.size());
            for (uint32_t v: nbrs) {
                if (!node_map.count(v)) continue;
                float d = distance_functions_t::compute(
                        metric, node_map.at(v).vector,
                        node_map.at(owner).vector, dims);
                cand.emplace_back(v, d);
            }
            std::sort(cand.begin(), cand.end(), vamana_candidate_t::max_cmp());
            robust_prune(owner, cand, 1.2f);
        }
    };

    /*--------------------------------------------------------------*/
    /* 3. Approximate in‑neighbors of p                             */
    /*--------------------------------------------------------------*/
    std::vector<uint32_t> approx_in;
    {
        for (const auto& nn: sr.nearest_nodes) {
            auto nit = node_map.find(nn.id);
            if (nit == node_map.end()) continue;
            if (std::find(nit->second.neighbors.begin(),
                          nit->second.neighbors.end(),
                          id) != nit->second.neighbors.end())
                approx_in.push_back(nn.id);
        }
    }

    /* fast exit: isolated node? */
    if (approx_in.empty() && it->second.neighbors.empty()) {
        medoid_tracker.sub(it->second.vector);
        delete_list.insert(id);
        node_map.erase(it);
        if (start_node == id) {                           // keep start live
            if (!node_map.empty())
                start_node = node_map.begin()->first;
        }
        return;
    }

    std::vector<uint32_t> scratch;   // reused for select_top_c()

    /*--------------------------------------------------------------*/
    /* 4. Patch in‑neighbors                                        */
    /*--------------------------------------------------------------*/
    for (uint32_t z: approx_in) {
        if (!node_map.count(z)) continue;
        select_top_c(z, scratch);
        patch_edges(z, scratch);
    }

    /*--------------------------------------------------------------*/
    /* 5. Patch outgoing neighbors of p                             */
    /*--------------------------------------------------------------*/
    for (uint32_t w: it->second.neighbors) {
        if (!node_map.count(w)) continue;
        select_top_c(w, scratch);
        if (scratch.empty()) continue;
        for (uint32_t y: scratch) {
            if (y == w || !node_map.count(y)) continue;
            patch_edges(y, {w});
        }
    }

    /*--------------------------------------------------------------*/
    /* 6. Book‑keeping                                              */
    /*--------------------------------------------------------------*/
    medoid_tracker.sub(it->second.vector);         // keep Σx correct
    delete_list.insert(id);                        // tomb‑stone
    node_map.erase(it);                            // physical erase

    /* guarantee a live start node */
    if (start_node == id && !node_map.empty()) {
        // quick & cheap: use (updated) centroid to pick a new medoid
        std::vector<float> centroid = medoid_tracker.centroid();
        search_result_t medoid_res;
        greedy_search(node_map.begin()->first, centroid, 1, 64, medoid_res);
        if (!medoid_res.nearest_nodes.empty()) {
            start_node = medoid_res.nearest_nodes.front().id;
        }
        else {
            start_node = node_map.begin()->first;
        }
    }
}

void Vamana::try_medoid_compute(const std::vector<float>& point,
                                bool force /* = false */) {
    medoid_tracker.add(point);

    if (force || medoid_tracker.should_recompute()) {
        std::vector<float> centroid = medoid_tracker.centroid();
        search_result_t res;
        /* small L keeps the overhead negligible */
        greedy_search(start_node, centroid, /*k=*/1, /*L=*/64, res);
        if (!res.nearest_nodes.empty())
            start_node.store(res.nearest_nodes[0].id,
                             std::memory_order_relaxed);
    }
}

uint32_t Vamana::get_start_node() {
    return start_node;
}

bool Vamana::validate_graph() {
    for (const auto& [node_id, node]: node_map) {
        std::unordered_set<uint32_t> unique_neighbors;
        for (uint32_t neighbor: node.neighbors) {
            if (unique_neighbors.count(neighbor) > 0 || unique_neighbors.size() > R) {
                //LOG(ERROR) << "Node " << node_id << " has duplicate neighbor " << neighbor << std::endl;
                return false;
            }

            unique_neighbors.insert(neighbor);
        }
    }
    return true;
}

vamana_node_t Vamana::get_node(uint32_t node_id) {
    vamana_node_t empty_node(R, {});
    if (delete_list.find(node_id) != delete_list.end()) {
        return empty_node;
    }

    auto it = node_map.find(node_id);
    return (it == node_map.end()) ? empty_node : it->second;
}

size_t Vamana::get_size() {
    return node_map.size();
}
