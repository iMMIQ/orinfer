/* CPU-only ACL ABI fixture. Copies remain queued until a stream/event fence. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { void *dst; const void *src; size_t n; int value, kind; void *stream; void *event; } Job;
typedef struct { void *stream; int ready, capture_supported; } Event;
static Job jobs[1024];
static int count, counters[10], failure, stream_creates, capturing;
static void *current;
static char aot_contract[4096];
void test_kernel_contract(const char *text) {
    if (strlen(text) >= sizeof(aot_contract)) abort();
    strcpy(aot_contract, text);
}
const char *orinfer_ascend_kernel_spec_v1(void) { return aot_contract; }
int orinfer_ascend_kernel_v1(void *const *buffers, uint64_t size, void *stream) {
    if (!buffers || size != 2 || !stream) return 7;
    memcpy(buffers[1], buffers[0], 4);
    return 0;
}
int orinfer_ascend_kernel_v2(void *const *buffers, uint64_t size, uint64_t rows, void *stream) {
    if (!buffers || size != 2 || !stream || !rows || rows > 4) return 7;
    memcpy(buffers[1], buffers[0], rows);
    return 0;
}
static int flush(void *stream) {
    if (failure == 3) return 7;
    for (int i=0;i<count;i++) {
        Job *j=&jobs[i];
        if (j->kind==0 || j->stream!=stream) continue;
        if (j->kind==1) memcpy(j->dst,j->src,j->n);
        if (j->kind==2) memset(j->dst,j->value,j->n);
        if (j->kind==3) ((Event*)j->event)->ready=1;
        if (j->kind==4 && !((Event*)j->event)->ready) {
            int status=flush(((Event*)j->event)->stream); if (status) return status;
        }
        j->kind=0;
    }
    return 0;
}
int test_counter(int n) { return counters[n]; }
void test_failure(int n) { failure=n; }
int aclInit(const char *p) { (void)p;counters[5]++;return 0; }
int aclFinalize(void) { counters[6]++;return 0; }
int aclrtGetDeviceCount(uint32_t *n) { *n=8;return 0; }
const char *aclrtGetSocName(void) { return "Ascend910B3"; }
int aclrtCreateContext(void **p,int device) { *p=malloc(1);*(uint8_t*)*p=(uint8_t)device;current=*p;counters[4]++;return 0; }
int aclrtDestroyContext(void *p) { if(current==p)current=NULL;free(p);counters[4]--;return 0; }
int aclrtGetCurrentContext(void **p) { *p=current;return current?0:107002; }
int aclrtSetCurrentContext(void *p) { current=p;return 0; }
int aclrtCreateStream(void **p) {
    if(failure==5 && ++stream_creates==2){*p=(void*)1;return 7;}
    *p=malloc(1);counters[3]++;return 0;
}
int aclrtDestroyStream(void *p) { free(p);counters[3]--;return 0; }
int aclrtSynchronizeStreamWithTimeout(void *s,int timeout) { (void)timeout;return flush(s); }
int aclrtCreateEventWithFlag(void **p,uint32_t flag) {
    if(flag!=1)return 7;
    if(failure==13 && counters[2]>0)return 7;
    *p=calloc(1,sizeof(Event));counters[2]++;return 0;
}
int aclrtDestroyEvent(void *p) { free(p);counters[2]--;return 0; }
int aclrtCreateEventExWithFlag(void **p,uint32_t flag) {
    int status=aclrtCreateEventWithFlag(p,flag);
    if(!status)((Event*)*p)->capture_supported=1;
    return status;
}
int aclrtRecordEvent(void *p,void *s) {
    if(failure==4)return 7;
    if(capturing && !((Event*)p)->capture_supported)return 207000;
    ((Event*)p)->stream=s;
    jobs[count++]=(Job){.kind=3,.stream=s,.event=p};return 0;
}
/* With no waiters, wait-status can complete while the record is pending. */
int aclrtQueryEventWaitStatus(void *p,int *state) { (void)p;*state=0;return 0; }
int aclrtQueryEventStatus(void *p,int *state) { *state=(failure==11 || ((Event*)p)->ready)?1:0;return 0; }
int aclrtSynchronizeEventWithTimeout(void *p,int timeout) { (void)timeout;return flush(((Event*)p)->stream); }
int aclrtStreamWaitEventWithTimeout(void *s,void *p,int timeout) {
    (void)timeout;jobs[count++]=(Job){.kind=4,.stream=s,.event=p};return 0;
}
int aclrtMalloc(void **p,size_t n,int policy) { (void)policy;*p=calloc(1,n);counters[0]++;return 0; }
int aclrtFree(void *p) { free(p);counters[0]--;return 0; }
int aclrtMallocHost(void **p,size_t n) { *p=calloc(1,n);counters[1]++;counters[9]++;return 0; }
int aclrtFreeHost(void *p) { free(p);counters[1]--;return 0; }
int aclrtMemcpyAsync(void *dst,size_t max,const void *src,size_t n,int kind,void *stream) {
    (void)kind;if(n>max)return 7;
    jobs[count++]=(Job){.dst=dst,.src=src,.n=n,.kind=1,.stream=stream};return 0;
}
int aclrtMemsetAsync(void *dst,size_t max,int value,size_t n,void *stream) {
    if(n>max)return 7;
    jobs[count++]=(Job){.dst=dst,.n=n,.value=value,.kind=2,.stream=stream};return 0;
}
int aclrtGetMemInfo(int kind,size_t *free,size_t *total) { (void)kind;*free=1<<28;*total=1<<29;return 0; }
int aclmdlRICaptureBegin(void *stream,int mode) { (void)stream;(void)mode;if(failure==6)return 7;capturing=1;return 0; }
int aclmdlRICaptureEnd(void *stream,void **model) { (void)stream;capturing=0;*model=malloc(1);counters[7]++;return 0; }
int aclmdlRIExecuteAsync(void *model,void *stream) { (void)model;(void)stream;return 0; }
int aclmdlRIDestroy(void *model) { free(model);counters[7]--;return 0; }
int HcclGetRootInfo(void *p) {if(!current)return 15;memset(p,0,4108);*(uint8_t*)p=*(uint8_t*)current;return 0;}
int HcclCommInitRootInfo(uint32_t ranks,const void *root,uint32_t rank,void **comm) {
    if(failure==10){*comm=(void*)1;return 7;}
    (void)ranks;(void)root;(void)rank;*comm=malloc(1);counters[8]++;return 0;
}
int HcclCommDestroy(void *comm) {free(comm);counters[8]--;return 0;}
int HcclGetCommAsyncError(void *comm,int *status) {(void)comm;*status=0;return 0;}
static size_t hccl_bytes(int type) {return type==0?1:type==11?2:(type==5||type==6)?8:4;}
int HcclAllReduce(void *source,void *destination,uint64_t n,int type,int op,void *comm,void *stream) {
    (void)op;(void)comm;size_t bytes=n*hccl_bytes(type);
    return aclrtMemcpyAsync(destination,bytes,source,bytes,3,stream);
}
int HcclReduceScatter(void *source,void *destination,uint64_t n,int type,int op,void *comm,void *stream) {
    return HcclAllReduce(source,destination,n,type,op,comm,stream);
}
int HcclAllGather(void *source,void *destination,uint64_t n,int type,void *comm,void *stream) {
    (void)comm;size_t bytes=n*hccl_bytes(type);
    jobs[count++]=(Job){.dst=destination,.src=source,.n=bytes,.kind=1,.stream=stream};return 0;
}
int HcclAlltoAllV(const void *source,const void *counts,const void *offsets,int type,const void *destination,
                 const void *receive,const void *displacements,int receive_type,void *comm,void *stream) {
    (void)offsets;(void)receive;(void)displacements;(void)receive_type;(void)comm;
    size_t bytes=(*(const uint64_t*)counts)*(type==0?1:type==5?8:4);
    return aclrtMemcpyAsync((void*)destination,bytes,source,bytes,3,stream);
}
