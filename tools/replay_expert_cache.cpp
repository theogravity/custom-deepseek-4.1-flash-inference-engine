// Replay per-GPU dispatches from the experiment/expert-cache-replay trace build.
// Arguments: gpu-N.bin, whitespace-separated inclusive wave start/end times in us.
#ifdef NDEBUG
#error "Replay requires enabled trace integrity checks"
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <vector>

struct Ref {
    unsigned key, capacity, before_group, group, wave;
    bool first, missed;
};
constexpr unsigned keys = 43 * 384;
constexpr int never = std::numeric_limits<int>::max();

std::vector<Ref> read_trace(const char *name, const char *windows) {
    std::ifstream waves(windows);
    std::vector<std::pair<uint64_t, uint64_t>> times;
    uint64_t lo, hi;
    while (waves >> lo >> hi)
        times.emplace_back(lo, hi);
    assert(times.size() == 7);
    std::ifstream in(name, std::ios::binary);
    assert(in);
    auto read = [&](unsigned bytes) {
        uint64_t v = 0;
        for (unsigned i = 0; i < bytes; ++i) {
            int c = in.get();
            assert(c != EOF);
            v |= uint64_t(c) << (8 * i);
        }
        return v;
    };
    std::vector<Ref> refs;
    unsigned wave = 0, group = 0;
    while (in.peek() != EOF) {
        auto ts = read(8);
        while (wave < times.size() && ts > times[wave].second)
            ++wave;
        assert(wave < times.size() && ts >= times[wave].first);
        unsigned before = read(2), cap = read(2), layer = read(1), n = read(1);
        assert(layer < 43 && n > 0 && n <= 64 && n <= cap && before <= cap);
        unsigned occupied = before;
        for (unsigned j = 0; j < n; ++j) {
            unsigned encoded = read(2), expert = encoded & 32767;
            bool miss = encoded >> 15;
            assert(expert < (layer < 40 ? 384u : 128u));
            // Alternative policies may never allocate beyond the observed LRU occupancy.
            occupied = std::min(cap, occupied + unsigned(miss));
            refs.push_back({layer * 384 + expert, occupied, before, group, wave, j == 0, miss});
        }
        ++group;
    }
    return refs;
}

enum class Policy { lru, segmented, frequency, aged_frequency, oracle };
using Order = std::tuple<int64_t, int, unsigned>;
struct Entry {
    Order order{};
    bool resident = false;
    int last_group = -1;
    int64_t frequency = 0;
};
struct Stats {
    unsigned refs = 0, misses = 0;
    std::array<unsigned, 2> by_kind{};
};

void replay(const std::vector<Ref> &refs, const std::vector<int> &next, Policy policy,
            const char *name, unsigned protected_percent = 0) {
    std::vector<Entry> entries(keys);
    std::set<Order> order;
    std::array<Stats, 7> counts{};
    unsigned hot = 0;
    int64_t age = 0;
    auto remove = [&](int group) {
        auto victim = order.begin();
        while (victim != order.end() && entries[std::get<2>(*victim)].last_group == group)
            ++victim;
        assert(victim != order.end());
        Entry &e = entries[std::get<2>(*victim)];
        if (policy == Policy::segmented && std::get<0>(*victim) == 1)
            --hot;
        if (policy == Policy::aged_frequency)
            age = std::max(age, std::get<0>(*victim));
        e.resident = false;
        order.erase(victim);
    };
    for (int i = 0; i < int(refs.size()); ++i) {
        auto r = refs[i];
        if (r.first)
            while (order.size() > r.before_group)
                remove(-2);
        Entry &e = entries[r.key];
        bool hit = e.resident;
        auto &tally = counts[r.wave];
        ++tally.refs;
        if (policy == Policy::lru)
            assert(hit != r.missed);
        int64_t priority = 0;
        if (hit)
            order.erase(e.order);
        else {
            ++tally.misses;
            ++tally.by_kind[r.key / 384 >= 40];
            while (order.size() >= r.capacity)
                remove(r.group);
        }
        switch (policy) {
        case Policy::lru:
            break;
        case Policy::frequency:
            priority = ++e.frequency;
            break;
        case Policy::aged_frequency:
            e.frequency = hit ? e.frequency + 1 : 1;
            priority = age + e.frequency;
            break;
        case Policy::oracle:
            priority = -int64_t(next[i]);
            break;
        case Policy::segmented:
            if (hit) {
                priority = 1;
                if (!std::get<0>(e.order))
                    ++hot;
            }
            break;
        }
        e.order = {priority, i, r.key};
        e.resident = true;
        e.last_group = r.group;
        order.insert(e.order);
        if (policy == Policy::segmented) {
            while (hot > r.capacity * protected_percent / 100) {
                auto oldest = order.lower_bound({1, 0, 0});
                assert(oldest != order.end());
                auto &demoted = entries[std::get<2>(*oldest)];
                order.erase(oldest);
                std::get<0>(demoted.order) = 0;
                // A demoted entry becomes newest in probation, with the key breaking a time tie.
                std::get<1>(demoted.order) = i;
                order.insert(demoted.order);
                --hot;
            }
        }
    }
    for (unsigned w = 0; w < counts.size(); ++w) {
        auto s = counts[w];
        std::cout << name << ',' << w << ',' << s.refs << ',' << s.misses << ',' << s.by_kind[0]
                  << ',' << s.by_kind[1] << '\n';
    }
}

// The Fenwick tree has one marker per previously seen key, at its most recent
// access. Keys newer than the previous use give the distinct-key reuse distance.
void reuse_distances(const std::vector<Ref> &refs) {
    std::vector<int> tree(refs.size() + 2), last(keys);
    auto add = [&](int i, int delta) {
        while (i < int(tree.size())) {
            tree[i] += delta;
            i += i & -i;
        }
    };
    auto prefix = [&](int i) {
        int n = 0;
        while (i > 0) {
            n += tree[i];
            i -= i & -i;
        }
        return n;
    };
    int distinct = 0;
    for (int i = 1; i <= int(refs.size()); ++i) {
        auto r = refs[i - 1];
        const int prev = last[r.key];
        if (r.missed) {
            int distance = prev ? distinct - prefix(prev) : -1;
            std::cout << r.wave << ',' << r.capacity << ',' << distance << '\n';
        }
        if (prev)
            add(prev, -1);
        else
            ++distinct;
        add(i, 1);
        last[r.key] = i;
    }
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4)
        return 2;
    auto refs = read_trace(argv[1], argv[2]);
    assert(!refs.empty() && refs.size() < unsigned(never));
    if (argc == 4) {
        if (std::string(argv[3]) != "--reuse-distances")
            return 2;
        reuse_distances(refs);
        return 0;
    }
    std::vector<int> future(keys, never), next(refs.size());
    for (int i = int(refs.size()) - 1; i >= 0; --i) {
        unsigned k = refs[i].key;
        next[i] = future[k];
        future[k] = i;
    }
    replay(refs, next, Policy::lru, "LRU");
    replay(refs, next, Policy::segmented, "SLRU25", 25);
    replay(refs, next, Policy::segmented, "SLRU50", 50);
    replay(refs, next, Policy::segmented, "SLRU75", 75);
    replay(refs, next, Policy::frequency, "LFU");
    replay(refs, next, Policy::aged_frequency, "LFUDA");
    replay(refs, next, Policy::oracle, "farthest-next-use");
}
