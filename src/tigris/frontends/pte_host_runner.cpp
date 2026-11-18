/*
 * Runner for ExecuTorch Cortex-M programs (.pte), used by `tigris run`,
 * which builds it for the host as below, and by scripts/pte_host_check.py.
 *
 * usage: pte_host_runner model.pte input.bin output.bin [input_after.bin [poison]]
 *
 * Loads the program with internal-consistency verification and, for each
 * input-sized slice of input.bin, fills every planned buffer with the poison
 * byte (default 0xA5) so that a read of a byte the inference has not written
 * sees neither a previous inference's value nor zero, sets the slice as method
 * input 0, executes "forward" and appends the raw bytes of every method output
 * to output.bin. With input_after.bin it also appends the bytes of method
 * input 0 as they are after the inference, read where the method keeps them.
 * Prints the number of inferences to stderr.
 *
 * Host build: an ExecuTorch 1.6.0 source tree (with its submodules) with the
 * cortex_m operators on CMSIS-NN v8.0.0, whose kernels compile to their
 * portable C paths on x86. The cortex_m build fetches CMSIS-NN at its pinned
 * tag unless -DCMSIS_NN_LOCAL_PATH names a checkout; the codegen step needs a
 * Python with ExecuTorch's build requirements (-DPYTHON_EXECUTABLE).
 *
 *   cmake -S executorch -B et-host -G Ninja -DCMAKE_BUILD_TYPE=Release \
 *     -DEXECUTORCH_BUILD_CORTEX_M=ON -DEXECUTORCH_BUILD_PORTABLE_OPS=OFF \
 *     -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF \
 *     -DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=OFF \
 *     -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON -DEXECUTORCH_ENABLE_LOGGING=ON
 *   cmake --build et-host
 *   B=et-host
 *   g++ -std=c++17 -O2 -I. -Iexecutorch/runtime/core/portable_type/c10 \
 *     -DC10_USING_CUSTOM_GENERATED_MACROS pte_host_runner.cpp \
 *     -Wl,--whole-archive $B/libexecutorch.a $B/backends/cortex_m/libcortex_m_ops_lib.a \
 *     -Wl,--no-whole-archive $B/libexecutorch_core.a $B/backends/cortex_m/libcortex_m_kernels.a \
 *     $B/backends/cortex_m/cmsis_nn-build/libcmsis-nn.a \
 *     $B/kernels/portable/cpu/util/libkernels_util_all_deps.a $B/libexecutorch_core.a \
 *     -lpthread -ldl -o pte_host_runner
 *
 * The g++ line runs from the directory that holds the executorch tree, with
 * this file's path in place of pte_host_runner.cpp.
 *
 * MVE kernels have no host path, so programs lowered for them run on an
 * emulated Cortex-M55 (QEMU's MPS3 AN547 board). The bare-metal libraries:
 *
 *   cmake -S executorch -B et-m55 \
 *     -DCMAKE_TOOLCHAIN_FILE=executorch/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake \
 *     -DEXECUTORCH_BUILD_PRESET_FILE=executorch/tools/cmake/preset/arm_baremetal.cmake \
 *     -DTARGET_CPU=cortex-m55
 *   cmake --build et-m55 --target executorch executorch_core cortex_m_ops_lib \
 *     cortex_m_kernels cmsis-nn kernels_util_all_deps
 *
 * then this file linked as on the host, with arm-none-eabi-g++ -mcpu=cortex-m55
 * -mfloat-abi=hard -mthumb -fno-exceptions -fno-rtti --specs=rdimon.specs
 * -nostartfiles, B=et-m55 and -lrdimon in place of -lpthread -ldl, plus a
 * startup file and a linker script. The linker script puts data, bss and heap
 * in the board's DDR at 0x60000000 (16 MB is enough). The startup file holds
 * the vector table and a reset handler that enables the FPU, zeroes .bss, calls
 * initialise_monitor_handles() and __libc_init_array(), reads the semihosting
 * command line (SYS_GET_CMDLINE, 0x15), splits it on spaces into argv and
 * exits with main's status; it also defines the et_pal_* platform functions
 * (logging to stderr, abort through exit). It runs as
 *
 *   qemu-system-arm -M mps3-an547 -nographic -monitor none -serial none \
 *     -kernel pte_m55_runner.elf -semihosting-config \
 *     enable=on,target=native,arg=runner,arg=model.pte,arg=input.bin,...
 *
 * with one arg= per runner argument; files are read and written on the host
 * through semihosting and the exit status is the runner's. A one-line wrapper
 * script around that command is the --runner of pte_host_check.py.
 */
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

// newlib's printf on bare metal may lack %zu.
static unsigned long ul(size_t n) { return static_cast<unsigned long>(n); }

static int fail(const char* what, Error e) {
  fprintf(stderr, "%s failed: 0x%" PRIx32 "\n", what, static_cast<uint32_t>(e));
  return 1;
}

// The size of an open file, leaving it at its start; -1 on failure.
static long file_size(FILE* f) {
  if (fseek(f, 0, SEEK_END) != 0) return -1;
  const long n = ftell(f);
  return fseek(f, 0, SEEK_SET) == 0 ? n : -1;
}

// The whole file in a 16-byte aligned buffer, the alignment Program::load
// may require of segments; nullptr on failure.
static uint8_t* read_aligned(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) return nullptr;
  const long n = file_size(f);
  uint8_t* data = n < 0 ? nullptr
                        : static_cast<uint8_t*>(aligned_alloc(16, (n + 16) / 16 * 16));
  if (data && fread(data, 1, n, f) != static_cast<size_t>(n)) {
    free(data);
    data = nullptr;
  }
  fclose(f);
  *size = n < 0 ? 0 : static_cast<size_t>(n);
  return data;
}

// The bytes a planned input occupies: set_input copies a memory-planned input
// into its buffer, so two inputs that differ in every byte change exactly those.
static Error locate_input(Method& method, Tensor& in, std::vector<uint8_t>& slice,
                          std::vector<std::vector<uint8_t>>& planned, uint8_t** where) {
  memset(slice.data(), 0x00, slice.size());
  Error e = method.set_input(EValue(in), 0);
  if (e != Error::Ok) return e;
  std::vector<std::vector<uint8_t>> before = planned;
  memset(slice.data(), 0xFF, slice.size());
  e = method.set_input(EValue(in), 0);
  if (e != Error::Ok) return e;
  *where = slice.data();  // an input that is not planned shares the slice
  size_t changed = 0, run = 0;
  for (size_t b = 0; b < planned.size(); b++) {
    for (size_t i = 0; i < planned[b].size(); i++) {
      if (planned[b][i] == before[b][i]) continue;
      if (changed++ == 0) {
        *where = planned[b].data() + i;
        while (i + run < planned[b].size() && run < slice.size() &&
               planned[b][i + run] != before[b][i + run])
          run++;
      }
    }
  }
  if (changed != 0 && (changed != slice.size() || run != slice.size())) {
    fprintf(stderr, "input 0 changed %lu planned bytes, not %lu in one run\n", ul(changed),
            ul(slice.size()));
    return Error::Internal;
  }
  return Error::Ok;
}

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    fprintf(stderr, "usage: %s model.pte input.bin output.bin [input_after.bin [poison]]\n",
            argv[0]);
    return 2;
  }
  unsigned long poison = 0xA5;
  if (argc == 6) {
    char* end = nullptr;
    poison = strtoul(argv[5], &end, 0);
    if (*argv[5] == '\0' || *end != '\0' || poison > 0xFF) {
      fprintf(stderr, "poison must be a byte, not %s\n", argv[5]);
      return 2;
    }
  }
  executorch::runtime::runtime_init();
  size_t image_size = 0;
  uint8_t* image = read_aligned(argv[1], &image_size);
  if (!image) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 3;
  }
  BufferDataLoader loader(image, image_size);
  Result<Program> program =
      Program::load(&loader, Program::Verification::InternalConsistency);
  if (!program.ok()) return fail("Program::load", program.error());
  Result<MethodMeta> meta = program->method_meta("forward");
  if (!meta.ok()) return fail("method_meta", meta.error());
  if (meta->num_inputs() != 1) {
    fprintf(stderr, "the method takes %lu inputs, the runner feeds one\n", ul(meta->num_inputs()));
    return 4;
  }
  Result<TensorInfo> info = meta->input_tensor_meta(0);
  if (!info.ok()) return fail("input_tensor_meta", info.error());

  std::vector<std::vector<uint8_t>> planned;
  std::vector<Span<uint8_t>> spans;
  for (size_t i = 0; i < meta->num_memory_planned_buffers(); i++) {
    planned.emplace_back(meta->memory_planned_buffer_size(i).get(), static_cast<uint8_t>(poison));
  }
  for (auto& b : planned) spans.emplace_back(b.data(), b.size());
  static uint8_t method_pool[1 << 20];
  static uint8_t temp_pool[1 << 16];
  MemoryAllocator method_alloc(sizeof(method_pool), method_pool);
  MemoryAllocator temp_alloc(sizeof(temp_pool), temp_pool);
  HierarchicalAllocator planned_mem({spans.data(), spans.size()});
  MemoryManager mm(&method_alloc, &planned_mem, &temp_alloc);
  Result<Method> method = program->load_method("forward", &mm);
  if (!method.ok()) return fail("load_method", method.error());

  // A tensor over `slice` with the input's sizes, dim order and dtype.
  const auto sizes = info->sizes();
  const auto order = info->dim_order();
  std::vector<TensorImpl::SizesType> in_sizes(sizes.begin(), sizes.end());
  std::vector<TensorImpl::DimOrderType> in_order(order.begin(), order.end());
  std::vector<TensorImpl::StridesType> in_strides(sizes.size());
  TensorImpl::StridesType stride = 1;
  for (size_t i = order.size(); i-- > 0;) {
    in_strides[order[i]] = stride;
    stride *= in_sizes[order[i]];
  }
  const size_t in_bytes = info->nbytes();
  std::vector<uint8_t> slice(in_bytes);
  TensorImpl in_impl(info->scalar_type(), static_cast<ssize_t>(in_sizes.size()),
                     in_sizes.data(), slice.data(), in_order.data(), in_strides.data());
  Tensor in(&in_impl);
  uint8_t* in_now = nullptr;
  Error e = locate_input(*method, in, slice, planned, &in_now);
  if (e != Error::Ok) return fail("set_input", e);

  FILE* fi = fopen(argv[2], "rb");
  const long in_all = fi ? file_size(fi) : -1;
  if (in_all < 0) {
    fprintf(stderr, "cannot read %s\n", argv[2]);
    return 3;
  }
  if (in_bytes == 0 || static_cast<size_t>(in_all) % in_bytes != 0) {
    fprintf(stderr, "input %lu bytes does not divide %lu\n", ul(in_bytes),
            ul(static_cast<size_t>(in_all)));
    return 4;
  }
  FILE* fo = fopen(argv[3], "wb");
  FILE* fa = argc >= 5 ? fopen(argv[4], "wb") : nullptr;
  if (!fo || (argc >= 5 && !fa)) {
    fprintf(stderr, "cannot write %s\n", fo ? argv[4] : argv[3]);
    return 3;
  }
  const size_t n = static_cast<size_t>(in_all) / in_bytes;
  for (size_t k = 0; k < n; k++) {
    for (auto& b : planned) memset(b.data(), static_cast<int>(poison), b.size());
    if (fread(slice.data(), 1, in_bytes, fi) != in_bytes) {
      fprintf(stderr, "cannot read %s\n", argv[2]);
      return 3;
    }
    e = method->set_input(EValue(in), 0);
    if (e != Error::Ok) return fail("set_input", e);
    e = method->execute();
    if (e != Error::Ok) return fail("execute", e);
    for (size_t o = 0; o < method->outputs_size(); o++) {
      const EValue& value = method->get_output(o);
      if (!value.isTensor()) {
        fprintf(stderr, "output %lu is not a tensor\n", ul(o));
        return 4;
      }
      const Tensor out = value.toTensor();
      fwrite(out.const_data_ptr(), 1, out.nbytes(), fo);
    }
    if (fa) fwrite(in_now, 1, in_bytes, fa);
  }
  fclose(fi);
  fclose(fo);
  if (fa) fclose(fa);
  free(image);
  fprintf(stderr, "ran %lu inputs\n", ul(n));
  return 0;
}
