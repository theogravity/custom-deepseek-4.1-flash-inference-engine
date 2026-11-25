// SPDX-License-Identifier: Apache-2.0
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>

// Mirrors Gather in integration/dsv41_native_engram.py. All addresses and shape
// arguments are fixed at capture time. Only the pinned ID contents change at replay.
struct Gather {
    const std::int64_t* ids;
    const std::uint8_t* weights;
    const std::uint8_t* scales;
    std::uint8_t* staged_weights;
    std::uint8_t* staged_scales;
    std::int64_t rows;
    std::int64_t vocab_start;
    std::int64_t vocab_end;
    std::int64_t weight_bytes;
    std::int64_t scale_bytes;
    std::int64_t weight_offset;
    std::int64_t scale_offset;
    int weight_fd;
    int scale_fd;
};

extern "C" std::size_t dsv41_engram_gather_size() noexcept {
    return sizeof(Gather);
}

// This symbol is passed directly to cudaLaunchHostFunc. A host function must
// finish all reads/writes before returning and must never issue a CUDA call.
extern "C" void dsv41_engram_gather(void* data) noexcept {
    const Gather& g = *static_cast<const Gather*>(data);

    // Queue reads before taking a serial sequence of major faults on cold mmap
    // pages. Hints are best-effort: the mapping remains the source of truth.
    // Repeated IDs in one speculative batch only need one advisory read.
    // A collision may cause an extra hint. Reset every call for cold replays.
    std::uint64_t hinted[256]{};
    for (std::int64_t i = 0; i < g.rows; ++i) {
        const auto id = g.ids[i];
        if (id >= g.vocab_start && id < g.vocab_end) {
            const auto tag = static_cast<std::uint64_t>(id) + 1;
            const auto slot = (tag * 11400714819323198485ull) >> 56;
            if (hinted[slot] == tag) {
                continue;
            }
            hinted[slot] = tag;
            posix_fadvise(g.weight_fd, g.weight_offset + id * g.weight_bytes,
                          g.weight_bytes, POSIX_FADV_WILLNEED);
            posix_fadvise(g.scale_fd, g.scale_offset + id * g.scale_bytes,
                          g.scale_bytes, POSIX_FADV_WILLNEED);
        }
    }

    for (std::int64_t i = 0; i < g.rows; ++i) {
        const auto id = g.ids[i];
        auto* weights = g.staged_weights + i * g.weight_bytes;
        auto* scales = g.staged_scales + i * g.scale_bytes;
        if (id >= g.vocab_start && id < g.vocab_end) {
            std::memcpy(weights, g.weights + id * g.weight_bytes, g.weight_bytes);
            std::memcpy(scales, g.scales + id * g.scale_bytes, g.scale_bytes);
        } else {
            // Both packed nibbles represent zero. Zero scale yields +0 BF16.
            std::memset(weights, 0x88, g.weight_bytes);
            std::memset(scales, 0, g.scale_bytes);
        }
    }
}
