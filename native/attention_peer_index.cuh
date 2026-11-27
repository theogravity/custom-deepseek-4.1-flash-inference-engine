// Each peer runs full, unchanged index tiles in its own default stream.
// The existing MoE event link fences producer reads and returned owner outputs.
struct PeerEvents;
extern "C" int ds_moe_peer_begin(PeerEvents **link, int primary, int peer);
extern "C" int ds_moe_peer_end(PeerEvents *link);
extern "C" void ds_moe_peer_free(PeerEvents *link);

namespace {
constexpr int peer_index_unavailable = -1099;
constexpr size_t peer_index_limit = size_t(256) << 20;
constexpr size_t peer_index_granule = size_t(1) << 20;
size_t peer_index_round(size_t bytes) {
    return (bytes+peer_index_granule-1)/peer_index_granule*peer_index_granule;
}
struct IndexMirror {
    uint64_t identity=0, stamp=0;
    uint8_t *data=nullptr;
    size_t capacity=0, valid=0;
};
struct PeerIndexCache {
    int owner, peer;
    PeerEvents *events=nullptr;
    IndexMirror mirrors[3]{};
    uint64_t clock=0, uploaded=0, tiles=0;
    uint8_t *stage=nullptr;
    size_t stage_capacity=0, query_offset=0, weights_offset=0, picks_offset=0;
    IndexMirror *current=nullptr;
    bool active=false;

    size_t bytes() const {
        size_t n=stage_capacity;
        for (const auto &m:mirrors) n+=m.capacity;
        return n;
    }
    IndexMirror *slot(uint64_t id) {
        IndexMirror *oldest=mirrors;
        for (auto &m:mirrors) {
            if (m.identity==id) return &m;
            if (m.stamp<oldest->stamp) oldest=&m;
        }
        return oldest;
    }
};
struct IndexOwnerGuard {
    int owner;
    ~IndexOwnerGuard(){ cudaSetDevice(owner); }
};
size_t peer_stage_bytes(size_t workspace,int rows) {
    // The native selection scratch is already padded to 256 bytes. All three
    // tensors have a whole number of 128-byte cache lines per query row.
    return peer_index_round(workspace+size_t(rows)*(ih*idim*sizeof(float)+ih*sizeof(float)+top*sizeof(int)));
}
int optional_peer_status(cudaError_t status) {
    if (status==cudaErrorMemoryAllocation || status==cudaErrorPeerAccessUnsupported || status==cudaErrorTooManyPeers) {
        cudaGetLastError();
        return peer_index_unavailable;
    }
    return int(status);
}
int discard_mirror(IndexMirror &m) {
    if (m.data) {
        auto rc=cudaFree(m.data);
        if (rc!=cudaSuccess) return int(rc);
    }
    m=IndexMirror{};
    return 0;
}
}  // namespace

extern "C" int ds4_index_peer_new(int owner,int peer,PeerIndexCache **out) {
    int supported=0;
    TRY(cudaDeviceCanAccessPeer(&supported,peer,owner));
    if (!supported) return peer_index_unavailable;
    TRY(cudaDeviceCanAccessPeer(&supported,owner,peer));
    if (!supported) return peer_index_unavailable;
    auto *cache=new(std::nothrow) PeerIndexCache{owner,peer};
    if (!cache) return peer_index_unavailable;
    *out=cache;
    return 0;
}

extern "C" size_t ds4_index_peer_bytes(const PeerIndexCache *p){return p ? p->bytes() : 0;}
extern "C" void ds4_index_peer_stats(const PeerIndexCache *p,uint64_t *bytes,uint64_t *tiles){
    *bytes=p->uploaded; *tiles=p->tiles;
}

extern "C" size_t ds4_index_peer_growth(PeerIndexCache *p,uint64_t id,size_t keys,size_t work,int rows){
    auto *slot=p->slot(id);
    size_t stage=peer_stage_bytes(work,rows);
    size_t mirror=peer_index_round(keys);
    if (stage+mirror>peer_index_limit) return SIZE_MAX;
    return (stage>p->stage_capacity ? stage-p->stage_capacity:0) +
           (mirror>slot->capacity ? mirror-slot->capacity:0);
}

extern "C" int ds4_index_peer_finish(PeerIndexCache *p) {
    if (!p || !p->active) return 0;
    int rc=ds_moe_peer_end(p->events);
    p->active=false;
    if (rc && p->current) p->current->valid=0;
    p->current=nullptr;
    return rc;
}

extern "C" int ds4_index_peer_begin(PeerIndexCache *p,uint64_t id,const void *owner_keys,
        size_t keys,size_t work,int rows) {
    IndexOwnerGuard guard{p->owner};
    if (p->active) return int(cudaErrorInvalidValue);
    TRY(cudaSetDevice(p->peer));
    size_t stage=peer_stage_bytes(work,rows), mirror=peer_index_round(keys);
    if (stage+mirror>peer_index_limit) return peer_index_unavailable;
    auto *selected=p->slot(id);
    if (selected->identity!=id) {
        selected->identity=id;
        selected->valid=0;
    }
    // This limit covers all mirrors and staging on this GPU, across sessions.
    while (p->bytes()-p->stage_capacity-selected->capacity +
           std::max(stage,p->stage_capacity)+std::max(mirror,selected->capacity)>peer_index_limit) {
        IndexMirror *old=nullptr;
        for (auto &m:p->mirrors)
            if (&m!=selected && m.capacity && (!old || m.stamp<old->stamp)) old=&m;
        if (!old) return peer_index_unavailable;
        int rc=discard_mirror(*old); if (rc) return rc;
    }
    if (stage>p->stage_capacity) {
        if (p->stage) { TRY(cudaFree(p->stage)); p->stage=nullptr; p->stage_capacity=0; }
        auto rc=cudaMalloc(&p->stage,stage);
        if (rc!=cudaSuccess) return optional_peer_status(rc);
        p->stage_capacity=stage;
    }
    if (mirror>selected->capacity) {
        int rc=discard_mirror(*selected); if(rc) return rc;
        auto status=cudaMalloc(&selected->data,mirror);
        if (status!=cudaSuccess) return optional_peer_status(status);
        selected->capacity=mirror; selected->identity=id;
    }
    if (selected->valid>keys) selected->valid=0;
    p->query_offset=work;
    p->weights_offset=work+size_t(rows)*ih*idim*sizeof(float);
    p->picks_offset=p->weights_offset+size_t(rows)*ih*sizeof(float);
    int rc=ds_moe_peer_begin(&p->events,p->owner,p->peer);
    if (rc) return optional_peer_status(static_cast<cudaError_t>(rc));
    p->active=true;
    p->current=selected;
    selected->stamp=++p->clock;
    if (keys>selected->valid) {
        auto *src=static_cast<const uint8_t *>(static_cast<const Buffer *>(owner_keys)->ptr);
        auto status=cudaSetDevice(p->peer);
        if (status==cudaSuccess)
            status=cudaMemcpyPeerAsync(selected->data+selected->valid,p->peer,src+selected->valid,p->owner,
                                       keys-selected->valid,nullptr);
        if (status!=cudaSuccess) {
            selected->valid=0;
            int finish=ds4_index_peer_finish(p);
            return finish ? finish:optional_peer_status(status);
        }
        p->uploaded+=keys-selected->valid;
        selected->valid=keys;
    }
    return 0;
}

extern "C" int ds4_index_peer_tile(PeerIndexCache *p,const float *query,const float *weights,
        int width,int pos,int ratio,int rows,int *picks,int mode) {
    IndexOwnerGuard guard{p->owner};
    if (!p->active || !p->current) return int(cudaErrorInvalidValue);
    TRY(cudaSetDevice(p->peer));
    auto *q=reinterpret_cast<float *>(p->stage+p->query_offset);
    auto *w=reinterpret_cast<float *>(p->stage+p->weights_offset);
    auto *ids=reinterpret_cast<int *>(p->stage+p->picks_offset);
    auto status=cudaMemcpyPeerAsync(q,p->peer,query,p->owner,size_t(rows)*ih*idim*4,nullptr);
    if (status==cudaSuccess) status=cudaMemcpyPeerAsync(w,p->peer,weights,p->owner,size_t(rows)*ih*4,nullptr);
    if (status!=cudaSuccess) return optional_peer_status(status);
    int rc=select_index_data(p->peer,p->stage,p->current->data,q,w,nullptr,width,pos,ratio,rows,ids,nullptr,mode);
    if (rc) return optional_peer_status(static_cast<cudaError_t>(rc));
    status=cudaMemcpyPeerAsync(picks,p->owner,ids,p->peer,size_t(rows)*top*4,nullptr);
    if (status!=cudaSuccess) return optional_peer_status(status);
    ++p->tiles;
    return 0;
}

extern "C" void ds4_index_peer_free(PeerIndexCache *p) {
    if (!p) return;
    int prior=p->owner;
    cudaGetDevice(&prior);
    ds4_index_peer_finish(p);
    ds_moe_peer_free(p->events);
    if (cudaSetDevice(p->peer)==cudaSuccess) {
        for (auto &m:p->mirrors) if(m.data) cudaFree(m.data);
        if(p->stage) cudaFree(p->stage);
    }
    delete p;
    cudaSetDevice(prior);
}
