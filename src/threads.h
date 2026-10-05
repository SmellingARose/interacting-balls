// Minimal portable threads, locks, atomics and timing (pthreads on macOS/Linux, Win32 on Windows).
// br_run_threads runs one job on n workers that share its atomic work counter: a worker that cannot be started runs
// on the calling thread instead, so the job always completes.
#ifndef BR_THREADS_H
#define BR_THREADS_H
#include <stdlib.h>
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <windows.h>
  typedef volatile LONG br_atomic_int;
  static inline int br_atomic_fetch_add(br_atomic_int* a, int v) { return (int)InterlockedExchangeAdd(a, v); }
  // Logical processors in every processor group (GetSystemInfo counts only the caller's group: 64 at most).
  static inline int br_cpu_count(void) {
  #if _WIN32_WINNT >= 0x0601
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); if (n > 0) return (int)n;
  #endif
    SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
  }
  static inline double br_now(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart; }
  static inline void br_sleep_ms(int ms) { Sleep(ms); }
  typedef CRITICAL_SECTION br_mutex;
  static inline void br_mutex_init(br_mutex* m) { InitializeCriticalSection(m); }
  static inline void br_lock(br_mutex* m) { EnterCriticalSection(m); }
  static inline void br_unlock(br_mutex* m) { LeaveCriticalSection(m); }
  typedef struct { void (*fn)(void*); void* arg; } br_thunk;
  static DWORD WINAPI br_thread_main(LPVOID p) { br_thunk* t = (br_thunk*)p; t->fn(t->arg); return 0; }
  typedef struct { HANDLE h; br_thunk t; } br_thread;
  static inline int br_thread_start(br_thread* th, void (*fn)(void*), void* arg) { th->t.fn = fn; th->t.arg = arg; th->h = CreateThread(NULL, 0, br_thread_main, &th->t, 0, NULL); return th->h ? 0 : -1; }
  static inline void br_thread_join(br_thread* th) { if (!th->h) return; WaitForSingleObject(th->h, INFINITE); CloseHandle(th->h); th->h = NULL; }
  static inline void br_run_threads(int n, void (*fn)(void*), void* arg) {
    if (n < 1) n = 1;
    br_thread* t = (br_thread*)malloc(sizeof(br_thread) * n);
    if (!t) { fn(arg); return; }
  #if _WIN32_WINNT >= 0x0601
    WORD ng = GetActiveProcessorGroupCount(); DWORD all = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  #endif
    for (int i = 0; i < n; i++) {
      t[i].t.fn = fn; t[i].t.arg = arg; t[i].h = CreateThread(NULL, 0, br_thread_main, &t[i].t, CREATE_SUSPENDED, NULL);
      if (!t[i].h) { fn(arg); continue; }
  #if _WIN32_WINNT >= 0x0601
      if (ng > 1 && all > 0) {   // more than 64 logical processors: Windows 10 keeps a process in one group, so spread the workers
        DWORD k = (DWORD)i % all; WORD g = 0;
        while (g + 1 < ng && k >= GetActiveProcessorCount(g)) k -= GetActiveProcessorCount(g++);
        DWORD m = GetActiveProcessorCount(g); GROUP_AFFINITY ga = { 0 };
        ga.Group = g; ga.Mask = m >= 8 * sizeof(KAFFINITY) ? ~(KAFFINITY)0 : ((KAFFINITY)1 << m) - 1;
        SetThreadGroupAffinity(t[i].h, &ga, NULL);
      }
  #endif
      ResumeThread(t[i].h);
    }
    for (int i = 0; i < n; i++) br_thread_join(&t[i]);
    free(t);
  }
#else
  #include <pthread.h>
  #include <unistd.h>
  #include <time.h>
  #ifdef __linux__
    #include <sched.h>
  #endif
  typedef int br_atomic_int;
  static inline int br_atomic_fetch_add(br_atomic_int* a, int v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
  // Processors this process may run on (Linux: its affinity mask, e.g. taskset or a container's cpuset; CMake defines
  // _GNU_SOURCE there for CPU_COUNT), else every online processor.
  static inline int br_cpu_count(void) {
  #if defined(__linux__) && defined(CPU_COUNT)
    { cpu_set_t s; CPU_ZERO(&s); if (sched_getaffinity(0, sizeof s, &s) == 0) { int n = CPU_COUNT(&s); if (n > 0) return n; } }
  #endif
    long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 4;
  }
  static inline double br_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
  static inline void br_sleep_ms(int ms) { usleep((useconds_t)ms * 1000); }
  typedef pthread_mutex_t br_mutex;
  static inline void br_mutex_init(br_mutex* m) { pthread_mutex_init(m, NULL); }
  static inline void br_lock(br_mutex* m) { pthread_mutex_lock(m); }
  static inline void br_unlock(br_mutex* m) { pthread_mutex_unlock(m); }
  typedef struct { void (*fn)(void*); void* arg; } br_thunk;
  static void* br_thread_main(void* p) { br_thunk* t = (br_thunk*)p; t->fn(t->arg); return NULL; }
  typedef struct { pthread_t h; br_thunk t; int ok; } br_thread;
  static inline int br_thread_start(br_thread* th, void (*fn)(void*), void* arg) { th->t.fn = fn; th->t.arg = arg; th->ok = pthread_create(&th->h, NULL, br_thread_main, &th->t) == 0; return th->ok ? 0 : -1; }
  static inline void br_thread_join(br_thread* th) { if (th->ok) pthread_join(th->h, NULL); th->ok = 0; }
  static inline void br_run_threads(int n, void (*fn)(void*), void* arg) {
    if (n < 1) n = 1;
    br_thread* t = (br_thread*)malloc(sizeof(br_thread) * n);
    if (!t) { fn(arg); return; }
    for (int i = 0; i < n; i++) if (br_thread_start(&t[i], fn, arg)) fn(arg);
    for (int i = 0; i < n; i++) br_thread_join(&t[i]);
    free(t);
  }
#endif
#endif
