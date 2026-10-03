// Minimal OpenCL 1.2 declarations + a runtime loader. The GPU driver (AMD Adrenalin on Windows, ROCm/Mesa on Linux,
// the system framework on macOS) provides the OpenCL library; nothing needs to be installed to build or run.
#ifndef BR_CL_H
#define BR_CL_H
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #define CL_CALL __stdcall
#else
  #define CL_CALL
#endif

typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_ulong; typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type; typedef cl_bitfield cl_mem_flags; typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint cl_bool; typedef cl_uint cl_device_info; typedef cl_uint cl_program_build_info;
typedef intptr_t cl_context_properties;
typedef struct _cl_platform_id* cl_platform_id; typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context; typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem; typedef struct _cl_program* cl_program; typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_event* cl_event;

#define CL_SUCCESS 0
#define CL_FALSE 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)

typedef struct {
  cl_int (CL_CALL *GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
  cl_int (CL_CALL *GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
  cl_int (CL_CALL *GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
  cl_context (CL_CALL *CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
  cl_command_queue (CL_CALL *CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
  cl_program (CL_CALL *CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
  cl_int (CL_CALL *BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
  cl_int (CL_CALL *GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
  cl_kernel (CL_CALL *CreateKernel)(cl_program, const char*, cl_int*);
  cl_mem (CL_CALL *CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
  cl_int (CL_CALL *SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
  cl_int (CL_CALL *EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
  cl_int (CL_CALL *EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
  cl_int (CL_CALL *EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
  cl_int (CL_CALL *Finish)(cl_command_queue);
  cl_int (CL_CALL *ReleaseMemObject)(cl_mem);
  cl_int (CL_CALL *ReleaseKernel)(cl_kernel);
  cl_int (CL_CALL *ReleaseProgram)(cl_program);
  cl_int (CL_CALL *ReleaseCommandQueue)(cl_command_queue);
  cl_int (CL_CALL *ReleaseContext)(cl_context);
} BrCL;

extern BrCL CL;
int br_cl_load(void);   // 1 = OpenCL available
#endif
