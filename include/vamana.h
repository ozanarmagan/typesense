#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <distance_functions.h>
#include <atomic>
#include <mutex>

struct vamana_candidate_t {
    uint32_t id;
    float distance;

    vamana_candidate_t(uint32_t node, float distance) : id(node), distance(distance) {

    }

    vamana_candidate_t() {

    }

    struct min_cmp {
        bool operator()(const vamana_candidate_t& a, const vamana_candidate_t& b) const {
            return a.distance > b.distance;
        }
    };

    struct max_cmp {
        bool operator()(const vamana_candidate_t& a, const vamana_candidate_t& b) const {
            return a.distance < b.distance;
        }
    };
};

struct StreamingMedoid {
    explicit StreamingMedoid(size_t dims, size_t recompute_every = 0)
            : _sum(dims, 0.0f),
              _n(0),
              _interval(recompute_every ? recompute_every : 10'000ULL),
              _countdown(_interval) {}

    /* call on every insert */
    void add(const std::vector<float>& x) {
        std::lock_guard<std::mutex> lk(_mtx);
        for (size_t i = 0; i < _sum.size(); ++i) _sum[i] += x[i];
        ++_n;
        --_countdown;
    }

    /* call on every physical delete */
    void sub(const std::vector<float>& x) {
        std::lock_guard<std::mutex> lk(_mtx);
        for (size_t i = 0; i < _sum.size(); ++i) _sum[i] -= x[i];
        --_n;
        --_countdown;
    }

    /* returns true when a recalculation should be performed */
    bool should_recompute() const noexcept { return _countdown == 0; }

    /* produces the current centroid and resets the counter */
    std::vector<float> centroid() {
        std::lock_guard<std::mutex> lk(_mtx);
        std::vector<float> c(_sum.size());
        float scale = 1.0f / static_cast<float>(_n);
        for (size_t i = 0; i < c.size(); ++i) c[i] = _sum[i] * scale;
        _countdown = _interval;
        return c;
    }

private:
    std::vector<float> _sum;
    uint64_t _n;
    const uint64_t _interval;
    uint64_t _countdown;
    mutable std::mutex _mtx;
};

using medoid_tracker_t = StreamingMedoid;

struct search_result_t {
    std::vector<vamana_candidate_t> nearest_nodes;
};

struct vamana_node_t {
    std::vector<uint32_t> neighbors;
    std::vector<float> vector;

    explicit vamana_node_t(size_t R, size_t dims);

    explicit vamana_node_t(size_t R, const std::vector<float>& vector);
};

template<class T>
inline void dedup_vector(std::vector<T>& v) {
    std::unordered_set<T> seen;
    v.erase(std::remove_if(v.begin(), v.end(),
                           [&](const T& x) { return !seen.insert(x).second; }),
            v.end());
}

// Base class for search filtering: override operator() in derived classes.
struct VamanaFilterBase {
    virtual ~VamanaFilterBase() = default;
    virtual bool operator()(uint32_t id) = 0;
};

class Vamana {
private:
    const size_t R;
    distance_metric_t metric;
    const size_t dims;

    // Stores the vector value and neighbors of each node
    std::unordered_map<uint32_t, vamana_node_t> node_map;

    std::atomic<uint64_t> start_node = 0;
    medoid_tracker_t medoid_tracker;

private:

    // Temporarily delete list used for marking nodes as deleted.
    // It's also used in greedy_search to ignore deleted elements.
    std::unordered_set<uint32_t> delete_list;

    void update_neighbors(uint32_t id, const std::vector<float>& vec, float alpha);

public:

    using FilterFn = VamanaFilterBase*;

    Vamana(size_t R, distance_metric_t metric, size_t dims);

    void greedy_search(const uint32_t start, const std::vector<float>& query, const size_t k, const size_t L,
                       search_result_t& search_result,
                       FilterFn filter = nullptr);

    void robust_prune(uint32_t p, std::vector<vamana_candidate_t>& candidates, float alpha);

    void insert(uint32_t id, const std::vector<float>& vec, size_t L, float alpha);

    void update(uint32_t id, const std::vector<float>& new_vec, size_t L, float alpha);

    void remove(uint32_t id);

    void batch_delete();

    void try_medoid_compute(const std::vector<float>& point, bool force = false);

    const std::unordered_map<uint32_t, vamana_node_t>& get_node_map() const;

    uint32_t get_start_node();

    vamana_node_t get_node(uint32_t node_id);

    bool validate_graph();

    size_t get_size();
};

class VisitedSet {
public:
    explicit VisitedSet(size_t n = 0) : _tag(1), _flags(n, 0) {}

    inline void clear() noexcept {
        ++_tag;
        if (_tag == 0) {              // wrap‑around: rare (4 G queries)
            std::fill(_flags.begin(), _flags.end(), 0);
            _tag = 1;
        }
    }

    inline bool mark(uint32_t id) {          // returns true if *newly* visited
        ensure_capacity(id);
        if (_flags[id] == _tag) return false;
        _flags[id] = _tag;
        return true;
    }

private:
    uint32_t _tag;              // monotonically increasing epoch
    std::vector<uint32_t> _flags;            // one uint32 per node‑id

    inline void ensure_capacity(uint32_t id) {
        if (id >= _flags.size()) _flags.resize(id + 1, 0);
    }
};

/// Simple per‑thread object pool so we never allocate on the hot path.
class VisitedSetPool {
public:
    VisitedSet* acquire(size_t n) {
        if (_pool.empty()) return new VisitedSet(n);
        VisitedSet* v = _pool.back();
        _pool.pop_back();
        v->clear();
        return v;
    }

    void release(VisitedSet* v) { _pool.push_back(v); }

private:
    std::vector<VisitedSet*> _pool;
};

/// one pool per thread → no locks on the critical path
static thread_local VisitedSetPool g_visited_pool;
