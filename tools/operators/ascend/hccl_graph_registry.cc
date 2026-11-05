// LGPL-3.0-or-later. CANN ABI declarations retain their separate upstream license.
// CANN 9.2 uses device-local model IDs in this registry. Give each device its
// own unchanged native registry; native IDs, callbacks and cleanup stay native.
#include "hccl_graph_registry_abi.h"
#include <atomic>
#include <dlfcn.h>

namespace {
using GetDevice = int (*)(int*);
std::atomic<GetDevice> device_query{nullptr};
std::atomic<int> failure{0};

struct Registries {
    std::mutex mutex;
    std::unordered_map<int, hccl::AclgraphCallback*> devices;
};
// Process-lifetime, just like the vendor singleton. The installed HCOMM methods
// remove graph/communicator entries; these small registry owners remain loaded
// across ACL initialization cycles and cannot outlive their native code.
Registries& registries() {
    static auto* instances = new Registries;
    return *instances;
}
}

extern "C" __attribute__((visibility("default")))
int orinfer_hccl_graph_registry_configure(GetDevice query) {
    if (!query) return 1;
    device_query.store(query, std::memory_order_release);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int orinfer_hccl_graph_registry_status() {
    return failure.load(std::memory_order_acquire);
}

namespace hccl {
AclgraphCallback& AclgraphCallback::GetInstance() {
    auto query = device_query.load(std::memory_order_acquire);
    if (!query) {
        // Also permits native offline validation through LD_PRELOAD. The Rust
        // loader configures the exact retained runtime API before HCCL use.
        query = reinterpret_cast<GetDevice>(dlsym(RTLD_NEXT, "aclrtGetDevice"));
    }
    int device = -1;
    int status = query ? query(&device) : 107002;
    if (status || device < 0) {
        int unset = 0;
        failure.compare_exchange_strong(unset, status ? status : 107002);
        // Supply a valid object to unwind the native call. Rust checks the
        // sticky error before submission and after each collective and retires
        // the session; invalid-context work never shares a healthy registry.
        device = -1;
    }
    auto& owners = registries();
    std::lock_guard<std::mutex> lock(owners.mutex);
    auto& instance = owners.devices[device];
    if (!instance) instance = new AclgraphCallback;
    return *instance;
}
}
