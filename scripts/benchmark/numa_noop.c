/* LD_PRELOAD shim for the memory-node sensitivity runs of vLLM.
 *
 * vLLM's CPU worker (v1/worker/cpu_worker.py) calls torch.ops._C.init_cpu_memory_env([node of its first allowed
 * core]), which calls numa_set_membind() and numa_migrate_pages() and so replaces the memory policy that
 * `numactl --membind=N` gave the server. With the cores on socket 0 that is always node 0, i.e. remote memory
 * can never be requested. This shim turns those two libnuma calls into no-ops, so the policy set by numactl
 * stays in force. It is loaded only when the requested memory node differs from the cores' node
 * (scripts/benchmark/run_online.py). Build inside the vLLM image:
 *   apptainer exec <image>.sif gcc -shared -fPIC -O2 -o numa_noop.so numa_noop.c
 */
#include <stdio.h>

void numa_set_membind(void *mask) {
  (void)mask;
  fputs("numa_noop: numa_set_membind ignored (numactl policy kept)\n", stderr);
}

int numa_migrate_pages(int pid, void *from, void *to) {
  (void)pid; (void)from; (void)to;
  fputs("numa_noop: numa_migrate_pages ignored\n", stderr);
  return 0;
}
