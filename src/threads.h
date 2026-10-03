// Minimal portable threads, locks, atomics and timing (pthreads on macOS/Linux, Win32 on Windows).
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
  static inline int br_cpu_count(void) { SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors; }
  static inline double br_now(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart; }
  static inline void br_sleep_ms(int ms) { Sleep(ms); }
  typedef CRITICAL_SECTION br_mutex;
  static inline void br_mutex_init(br_mutex* m) { InitializeCriticalSection(m); }
  static inline void br_lock(br_mutex* m) { EnterCriticalSection(m); }
  static inline void br_unlock(br_mutex* m) { LeaveCriticalSection(m); }
  typedef struct { void (*fn)(void*); void* arg; } br_thunk;
  static DWORD WINAPI br_thread_main(LPVOID p) { br_thunk* t = (br_thunk*)p; t->fn(t->arg); return 0; }
  typedef struct { HANDLE h; br_thunk t; } br_thread;
  static inline void br_thread_start(br_thread* th, void (*fn)(void*), void* arg) { th->t.fn = fn; th->t.arg = arg; th->h = CreateThread(NULL, 0, br_thread_main, &th->t, 0, NULL); }
  static inline void br_thread_join(br_thread* th) { WaitForSingleObject(th->h, INFINITE); CloseHandle(th->h); }
  static inline void br_run_threads(int n, void (*fn)(void*), void* arg) {
    br_thread* t = (br_thread*)malloc(sizeof(br_thread) * n);
    for (int i = 0; i < n; i++) br_thread_start(&t[i], fn, arg);
    for (int i = 0; i < n; i++) br_thread_join(&t[i]);
    free(t);
  }
#else
  #include <pthread.h>
  #include <unistd.h>
  #include <time.h>
  typedef int br_atomic_int;
  static inline int br_atomic_fetch_add(br_atomic_int* a, int v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
  static inline int br_cpu_count(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 4; }
  static inline double br_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
  static inline void br_sleep_ms(int ms) { usleep((useconds_t)ms * 1000); }
  typedef pthread_mutex_t br_mutex;
  static inline void br_mutex_init(br_mutex* m) { pthread_mutex_init(m, NULL); }
  static inline void br_lock(br_mutex* m) { pthread_mutex_lock(m); }
  static inline void br_unlock(br_mutex* m) { pthread_mutex_unlock(m); }
  typedef struct { void (*fn)(void*); void* arg; } br_thunk;
  static void* br_thread_main(void* p) { br_thunk* t = (br_thunk*)p; t->fn(t->arg); return NULL; }
  typedef struct { pthread_t h; br_thunk t; } br_thread;
  static inline void br_thread_start(br_thread* th, void (*fn)(void*), void* arg) { th->t.fn = fn; th->t.arg = arg; pthread_create(&th->h, NULL, br_thread_main, &th->t); }
  static inline void br_thread_join(br_thread* th) { pthread_join(th->h, NULL); }
  static inline void br_run_threads(int n, void (*fn)(void*), void* arg) {
    br_thread* t = (br_thread*)malloc(sizeof(br_thread) * n);
    for (int i = 0; i < n; i++) br_thread_start(&t[i], fn, arg);
    for (int i = 0; i < n; i++) br_thread_join(&t[i]);
    free(t);
  }
#endif
#endif
