// Minimal native ABI fixture. The vendor-style singleton deliberately has no
// device key; the engine's adapter must isolate its unchanged native objects.
#include "hccl_graph_registry_abi.h"

namespace {
thread_local int device = 0;
}

namespace hccl {
AclgraphCallback& AclgraphCallback::GetInstance() {
    static AclgraphCallback original;
    return original;
}
AclgraphCallback::~AclgraphCallback() = default;

HcclResult AclgraphCallback::RegisterModelId(HcclCommunicator* owner, aclmdlRI, u64 model) {
    std::lock_guard<std::mutex> lock(resMutex_);
    captureResMap_[model][owner];
    return 0;
}
HcclResult AclgraphCallback::CleanCaptureRes(u64 model) {
    std::lock_guard<std::mutex> lock(resMutex_);
    auto found = captureResMap_.find(model);
    if (found == captureResMap_.end()) return 0;
    return static_cast<int>(reinterpret_cast<uintptr_t>(found->second.begin()->first));
}
void AclgraphCallback::CleanCaptureRes(HcclCommunicator* owner) {
    std::lock_guard<std::mutex> lock(resMutex_);
    for (auto it = captureResMap_.begin(); it != captureResMap_.end();) {
        it->second.erase(owner);
        if (it->second.empty()) it = captureResMap_.erase(it);
        else ++it;
    }
}
}

extern "C" int aclrtGetDevice(int* output) { *output = device; return 0; }
extern "C" int aclsysGetVersionNum(char*, int* output) { *output = 90200000; return 0; }
extern "C" void fixture_set_device(int value) { device = value; }
extern "C" int fixture_capture(u64 model, uintptr_t owner) {
    return hccl::AclgraphCallback::GetInstance().RegisterModelId(
        reinterpret_cast<hccl::HcclCommunicator*>(owner), nullptr, model);
}
extern "C" int fixture_owner(u64 model) {
    return hccl::AclgraphCallback::GetInstance().CleanCaptureRes(model);
}
extern "C" void fixture_destroy(uintptr_t owner) {
    hccl::AclgraphCallback::GetInstance().CleanCaptureRes(
        reinterpret_cast<hccl::HcclCommunicator*>(owner));
}
