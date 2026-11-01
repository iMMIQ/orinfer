/* Report whether a native module unloads after the owning ACL finalization. */
static int (*counter)(int);
static int *unloaded_after_finalize;

void test_aot_lifetime(int (*read_counter)(int), int *result) {
    counter = read_counter;
    unloaded_after_finalize = result;
}

__attribute__((destructor)) static void unload(void) {
    if (unloaded_after_finalize) *unloaded_after_finalize = counter(6);
}
