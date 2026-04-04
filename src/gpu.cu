#include "Random.h"
#include "gpu.h"

#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <random>

#define PANIC(...)                                                             \
  {                                                                            \
    std::fprintf(stderr, __VA_ARGS__);                                         \
    std::abort();                                                              \
  }

#define TRY_CUDA(expr) try_cuda(expr, __FILE__, __LINE__)

#define PREFILTER_THRESHOLD_1 -0.4f
#define PREFILTER_THRESHOLD_2 -0.6f

#define CENTRE_THRESHOLD_1 -0.2f
#define CENTRE_THRESHOLD_2 -0.4f

#define SMALL_PREFILTER_RADIUS 2048
#define FINAL_PREFILTER_RADIUS 512
#define ISLAND_SEARCH_RADIUS 256

#define SCORE_REQUIREMENT -7.5f
#define RING_RADIUS 640
#define RING_POINT_THRESHOLD -0.9f
#define RING_POINT_MIN_PASSED 15

#define ISLAND_THRESHOLD -0.19f

void try_cuda(cudaError_t error, const char *file, uint64_t line) {
  if (error == cudaSuccess)
    return;

  PANIC("CUDA error at %s:%" PRIu64 ": %s\n", file, line,
        cudaGetErrorString(error));
}

// from cubiomes
constexpr XrsrForkHash hash_continentalness{
    0x83886c9d0ae3a662, 0xafa638a61b42e8ad}; // md5 "minecraft:continentalness"
constexpr XrsrForkHash hash_continentalness_large{
    0x9a3f51a113fce8dc,
    0xee2dbd157e5dcdad}; // md5 "minecraft:continentalness_large"
constexpr XrsrForkHash hash_octave[]{
    {0xb198de63a8012672, 0x7b84cad43ef7b5a8}, // md5 "octave_-12"
    {0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8}, // md5 "octave_-11"
    {0x36d326eed40efeb2, 0x5be9ce18223c636a}, // md5 "octave_-10"
    {0x082fe255f8be6631, 0x4e96119e22dedc81}, // md5 "octave_-9"
    {0x0ef68ec68504005e, 0x48b6bf93a2789640}, // md5 "octave_-8"
    {0xf11268128982754f, 0x257a1d670430b0aa}, // md5 "octave_-7"
    {0xe51c98ce7d1de664, 0x5f9478a733040c45}, // md5 "octave_-6"
    {0x6d7b49e7e429850a, 0x2e3063c622a24777}, // md5 "octave_-5"
    {0xbd90d5377ba1b762, 0xc07317d419a7548d}, // md5 "octave_-4"
    {0x53d39c6752dac858, 0xbcd1c5a80ab65b3e}, // md5 "octave_-3"
    {0xb4a24d7a84e7677b, 0x023ff9668e89b5c4}, // md5 "octave_-2"
    {0xdffa22b534c5f608, 0xb9b67517d3665ca9}, // md5 "octave_-1"
    {0xd50708086cef4d7c, 0x6e1651ecc7f43309}, // md5 "octave_0"
};

struct ImprovedNoise {
  uint8_t p[256];
  float xo;
  float yo;
  float zo;
};

struct Octave {
  ImprovedNoise noise;
  double input_factor;
  double value_factor;
};

template <size_t N> struct NoiseParameters {
  int32_t first_octave;
  std::array<double, N> amplitudes;
};

template <size_t N>
constexpr NoiseParameters<N>
make_noise_parameters(int32_t first_octave, const double (&amplitudes)[N]) {
  std::array<double, N> amp{};
  std::copy(std::begin(amplitudes), std::end(amplitudes), amp.begin());
  return {first_octave, amp};
}

constexpr auto continentalness_parameters =
    make_noise_parameters(-9, {1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0});
constexpr auto continentalness_large_parameters =
    make_noise_parameters(-11, {1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0});

struct OctaveConfig {
  XrsrForkHash fork_hash;
  double input_factor;
  double value_factor;
};

template <size_t N> struct NormalNoiseConfig {
  XrsrForkHash fork_hash;
  std::array<OctaveConfig, N> octaves_a;
  std::array<OctaveConfig, N> octaves_b;
};

template <size_t N>
constexpr NormalNoiseConfig<N>
make_normal_noise_config(const NoiseParameters<N> &noise_parameters,
                         const XrsrForkHash &fork_hash) {
  NormalNoiseConfig<N> res{fork_hash};

  const auto first_octave = noise_parameters.first_octave;
  const auto &amplitudes = noise_parameters.amplitudes;

  double root_value_factor =
      0.16666666666666666 / (0.1 * (1.0 + 1.0 / amplitudes.size()));

  double input_factor = 1.0 / (1 << -first_octave);
  double value_factor = (1 << (amplitudes.size() - 1)) /
                        ((1 << amplitudes.size()) - 1.0) * root_value_factor;

  for (size_t i = 0; i < amplitudes.size(); i++) {
    res.octaves_a[i] = {hash_octave[first_octave + 12 + i], input_factor,
                        value_factor * amplitudes[i]};
    res.octaves_b[i] = {hash_octave[first_octave + 12 + i],
                        input_factor * 1.0181268882175227,
                        value_factor * amplitudes[i]};
    input_factor *= 2.0;
    value_factor *= 0.5;
  }

  return res;
}

constexpr auto continentalness_config =
    make_normal_noise_config(continentalness_parameters, hash_continentalness);
constexpr auto continentalness_large_config = make_normal_noise_config(
    continentalness_large_parameters, hash_continentalness_large);
constexpr auto chosen_continentalness_config =
    large_biomes ? continentalness_large_config : continentalness_config;
__device__ constexpr auto device_chosen_continentalness_config =
    chosen_continentalness_config;

// switch - 4.745 Gsps
// int8_t[3][16] - 5.293 Gsps
// float[3][16] - 5.324 Gsps
// uint32_t[16] - 5.306 Gsps

struct GradDotTable {
  float x[16];
  float y[16];
  float z[16];
};

__device__ GradDotTable device_grad_dot_table;

void init_grad_dot_table() {
  GradDotTable table;
  table.x[0] = 1;
  table.y[0] = 1;
  table.z[0] = 0; // { 1,  1,  0}
  table.x[1] = -1;
  table.y[1] = 1;
  table.z[1] = 0; // {-1,  1,  0}
  table.x[2] = 1;
  table.y[2] = -1;
  table.z[2] = 0; // { 1, -1,  0}
  table.x[3] = -1;
  table.y[3] = -1;
  table.z[3] = 0; // {-1, -1,  0}
  table.x[4] = 1;
  table.y[4] = 0;
  table.z[4] = 1; // { 1,  0,  1}
  table.x[5] = -1;
  table.y[5] = 0;
  table.z[5] = 1; // {-1,  0,  1}
  table.x[6] = 1;
  table.y[6] = 0;
  table.z[6] = -1; // { 1,  0, -1}
  table.x[7] = -1;
  table.y[7] = 0;
  table.z[7] = -1; // {-1,  0, -1}
  table.x[8] = 0;
  table.y[8] = 1;
  table.z[8] = 1; // { 0,  1,  1}
  table.x[9] = 0;
  table.y[9] = -1;
  table.z[9] = 1; // { 0, -1,  1}
  table.x[10] = 0;
  table.y[10] = 1;
  table.z[10] = -1; // { 0,  1, -1}
  table.x[11] = 0;
  table.y[11] = -1;
  table.z[11] = -1; // { 0, -1, -1}
  table.x[12] = 1;
  table.y[12] = 1;
  table.z[12] = 0; // { 1,  1,  0}
  table.x[13] = 0;
  table.y[13] = -1;
  table.z[13] = 1; // { 0, -1,  1}
  table.x[14] = -1;
  table.y[14] = 1;
  table.z[14] = 0; // {-1,  1,  0}
  table.x[15] = 0;
  table.y[15] = -1;
  table.z[15] = -1; // { 0, -1, -1}

  void *device_grad_dot_table_addr;
  TRY_CUDA(
      cudaGetSymbolAddress(&device_grad_dot_table_addr, device_grad_dot_table));
  TRY_CUDA(cudaMemcpy(device_grad_dot_table_addr, &table, sizeof(GradDotTable),
                      cudaMemcpyHostToDevice));
}

__device__ float gradDot(const GradDotTable &table, uint8_t p, float x, float y,
                         float z) {
  return x * table.x[p & 0xF] + y * table.y[p & 0xF] + z * table.z[p & 0xF];
}

__device__ float smoothstep(float value) {
  return value * value * value * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__device__ float lerp1(float fx, float v0, float v1) {
  return v0 + fx * (v1 - v0);
}

__device__ float lerp2(float fx, float fy, float v00, float v10, float v01,
                       float v11) {
  return lerp1(fy, lerp1(fx, v00, v10), lerp1(fx, v01, v11));
}

__device__ float lerp3(float fx, float fy, float fz, float v000, float v100,
                       float v010, float v110, float v001, float v101,
                       float v011, float v111) {
  return lerp1(fz, lerp2(fx, fy, v000, v100, v010, v110),
               lerp2(fx, fy, v001, v101, v011, v111));
}

__device__ float sample_noise(const GradDotTable &table,
                              const ImprovedNoise &noise, float x, float y,
                              float z) {
  x += noise.xo;
  y += noise.yo;
  z += noise.zo;
  float floor_x = std::floor(x);
  float floor_y = std::floor(y);
  float floor_z = std::floor(z);
  float frac_x = x - floor_x;
  float frac_y = y - floor_y;
  float frac_z = z - floor_z;
  int32_t int_x = floor_x;
  int32_t int_y = floor_y;
  int32_t int_z = floor_z;
  uint8_t p0 = noise.p[(int_x) & 0xFF];
  uint8_t p1 = noise.p[(int_x + 1) & 0xFF];
  uint8_t p00 = noise.p[(p0 + int_y) & 0xFF];
  uint8_t p01 = noise.p[(p0 + int_y + 1) & 0xFF];
  uint8_t p10 = noise.p[(p1 + int_y) & 0xFF];
  uint8_t p11 = noise.p[(p1 + int_y + 1) & 0xFF];
  float n000 =
      gradDot(table, noise.p[(p00 + int_z) & 0xFF], frac_x, frac_y, frac_z);
  float n100 = gradDot(table, noise.p[(p10 + int_z) & 0xFF], frac_x - 1.0f,
                       frac_y, frac_z);
  float n010 = gradDot(table, noise.p[(p01 + int_z) & 0xFF], frac_x,
                       frac_y - 1.0f, frac_z);
  float n110 = gradDot(table, noise.p[(p11 + int_z) & 0xFF], frac_x - 1.0f,
                       frac_y - 1.0f, frac_z);
  float n001 = gradDot(table, noise.p[(p00 + int_z + 1) & 0xFF], frac_x, frac_y,
                       frac_z - 1.0f);
  float n101 = gradDot(table, noise.p[(p10 + int_z + 1) & 0xFF], frac_x - 1.0f,
                       frac_y, frac_z - 1.0f);
  float n011 = gradDot(table, noise.p[(p01 + int_z + 1) & 0xFF], frac_x,
                       frac_y - 1.0f, frac_z - 1.0f);
  float n111 = gradDot(table, noise.p[(p11 + int_z + 1) & 0xFF], frac_x - 1.0f,
                       frac_y - 1.0f, frac_z - 1.0f);
  float fx = smoothstep(frac_x);
  float fy = smoothstep(frac_y);
  float fz = smoothstep(frac_z);
  return lerp3(fx, fy, fz, n000, n100, n010, n110, n001, n101, n011, n111);
}

__device__ float wrap(float value) {
  // return value - std::floor(value / 256.0) * 256.0;
  return value;
}

template <OctaveConfig config>
__device__ float sample_octave(const GradDotTable &table,
                               const ImprovedNoise &noise, int32_t x, int32_t y,
                               int32_t z) {
  return sample_noise(table, noise, wrap(x * (float)config.input_factor),
                      wrap(y * (float)config.input_factor),
                      wrap(z * (float)config.input_factor)) *
         (float)config.value_factor;
}

__device__ void init_noise(ImprovedNoise &noise, XrsrRandom &&random) {
  noise.xo = random.nextFloat() * 256.0f;
  noise.yo = random.nextFloat() * 256.0f;
  noise.zo = random.nextFloat() * 256.0f;

  for (uint32_t i = 0; i < 256; i++) {
    noise.p[i] = i;
  }
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t j = random.nextInt(256 - i);
    uint8_t b = noise.p[i];
    noise.p[i] = noise.p[i + j];
    noise.p[i + j] = b;
  }
}
struct DeviceBuffer {
  void *data;
  size_t size;

  DeviceBuffer(size_t size) : size(size) { TRY_CUDA(cudaMalloc(&data, size)); }

  ~DeviceBuffer() { TRY_CUDA(cudaFree(data)); }
};

template <typename T> struct OutputBuffer {
  T *data;
  uint32_t &len;
  uint32_t max_len;

  OutputBuffer(T *data, uint32_t &len, uint32_t max_len)
      : data(data), len(len), max_len(max_len) {}

  OutputBuffer(const DeviceBuffer &buffer, uint32_t &len)
      : data((T *)buffer.data), len(len), max_len(buffer.size / sizeof(T)) {}

  OutputBuffer(const OutputBuffer<T> &other)
      : data(other.data), len(other.len), max_len(other.max_len) {}
};

template <typename T> struct InputBuffer {
  const T *data;
  const uint32_t &len;

  InputBuffer(const T *data, const uint32_t &len) : data(data), len(len) {}

  InputBuffer(const OutputBuffer<T> &buffer)
      : data(buffer.data), len(buffer.len) {}

  InputBuffer(const InputBuffer<T> &other) : data(other.data), len(other.len) {}
};

namespace KernelSeed1 {
constexpr uint32_t threads_per_run = UINT64_C(1) << 16;
constexpr uint32_t threads_per_block = 32;

struct Result {
  ImprovedNoise continentalness_0A;
  ImprovedNoise continentalness_0B;
  ImprovedNoise continentalness_1A;
  ImprovedNoise continentalness_1B;
  ImprovedNoise continentalness_2A;
  ImprovedNoise continentalness_2B;
  ImprovedNoise continentalness_3A;
  ImprovedNoise continentalness_3B;
  ImprovedNoise continentalness_4A;
  ImprovedNoise continentalness_4B;
  ImprovedNoise continentalness_5A;
  ImprovedNoise continentalness_5B;
  ImprovedNoise continentalness_6A;
  ImprovedNoise continentalness_6B;
  ImprovedNoise continentalness_7A;
  ImprovedNoise continentalness_7B;
  ImprovedNoise continentalness_8A;
  ImprovedNoise continentalness_8B;
};

template <size_t Octaves> struct ResultSampler {
  ImprovedNoise octaves[Octaves];

  __device__ float sample(const GradDotTable &table, int32_t x, int32_t y,
                          int32_t z) const {
    float val = 0;
    if constexpr (Octaves >= 1)
      val += sample_octave<chosen_continentalness_config.octaves_a[0]>(
          table, octaves[0], x, y, z);
    if constexpr (Octaves >= 2)
      val += sample_octave<chosen_continentalness_config.octaves_b[0]>(
          table, octaves[1], x, y, z);
    if constexpr (Octaves >= 3)
      val += sample_octave<chosen_continentalness_config.octaves_a[1]>(
          table, octaves[2], x, y, z);
    if constexpr (Octaves >= 4)
      val += sample_octave<chosen_continentalness_config.octaves_b[1]>(
          table, octaves[3], x, y, z);
    if constexpr (Octaves >= 5)
      val += sample_octave<chosen_continentalness_config.octaves_a[2]>(
          table, octaves[4], x, y, z);
    if constexpr (Octaves >= 6)
      val += sample_octave<chosen_continentalness_config.octaves_b[2]>(
          table, octaves[5], x, y, z);
    if constexpr (Octaves >= 7)
      val += sample_octave<chosen_continentalness_config.octaves_a[3]>(
          table, octaves[6], x, y, z);
    if constexpr (Octaves >= 8)
      val += sample_octave<chosen_continentalness_config.octaves_b[3]>(
          table, octaves[7], x, y, z);
    if constexpr (Octaves >= 9)
      val += sample_octave<chosen_continentalness_config.octaves_a[4]>(
          table, octaves[8], x, y, z);
    if constexpr (Octaves >= 10)
      val += sample_octave<chosen_continentalness_config.octaves_b[4]>(
          table, octaves[9], x, y, z);
    if constexpr (Octaves >= 11)
      val += sample_octave<chosen_continentalness_config.octaves_a[5]>(
          table, octaves[10], x, y, z);
    if constexpr (Octaves >= 12)
      val += sample_octave<chosen_continentalness_config.octaves_b[5]>(
          table, octaves[11], x, y, z);
    if constexpr (Octaves >= 13)
      val += sample_octave<chosen_continentalness_config.octaves_a[6]>(
          table, octaves[12], x, y, z);
    if constexpr (Octaves >= 14)
      val += sample_octave<chosen_continentalness_config.octaves_b[6]>(
          table, octaves[13], x, y, z);
    if constexpr (Octaves >= 15)
      val += sample_octave<chosen_continentalness_config.octaves_a[7]>(
          table, octaves[14], x, y, z);
    if constexpr (Octaves >= 16)
      val += sample_octave<chosen_continentalness_config.octaves_b[7]>(
          table, octaves[15], x, y, z);
    if constexpr (Octaves >= 17)
      val += sample_octave<chosen_continentalness_config.octaves_a[8]>(
          table, octaves[16], x, y, z);
    if constexpr (Octaves >= 18)
      val += sample_octave<chosen_continentalness_config.octaves_b[8]>(
          table, octaves[17], x, y, z);
    return val;
  }
};

__device__ Result results[threads_per_run];
// constexpr size_t a = sizeof(results);

__device__ void copy_noise(ImprovedNoise (&shared_noise)[threads_per_block],
                           ImprovedNoise Result::*result_member) {
  for (uint32_t result_index = 0; result_index < threads_per_block;
       result_index++) {
    ImprovedNoise &src = shared_noise[result_index];
    ImprovedNoise &dst =
        results[blockIdx.x * blockDim.x + result_index].*result_member;
    for (uint32_t i = threadIdx.x; i < sizeof(ImprovedNoise) / sizeof(uint32_t);
         i += threads_per_block) {
      reinterpret_cast<uint32_t *>(&dst)[i] =
          reinterpret_cast<uint32_t *>(&src)[i];
    }
  }
}

__device__ void init_octave(const XrsrRandomFork &noise_fork,
                            const XrsrForkHash &fork_hash,
                            ImprovedNoise Result::*result_member) {
  __shared__ ImprovedNoise shared_noise[threads_per_block];

  init_noise(shared_noise[threadIdx.x], noise_fork.from(fork_hash));
  __syncthreads();
  copy_noise(shared_noise, result_member);
  __syncthreads();
}

__global__
__launch_bounds__(threads_per_block) void kernel(InputBuffer<uint64_t> input) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index / threads_per_block * threads_per_block >= input.len)
    return;
  uint64_t seed = input.data[index];

  const auto seed_fork = XrsrRandom(seed).fork();
  auto noise_random =
      seed_fork.from(device_chosen_continentalness_config.fork_hash);

  const auto noise_a_fork = noise_random.fork();
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[0].fork_hash,
              &Result::continentalness_0A);

  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[1].fork_hash,
              &Result::continentalness_1A);
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[2].fork_hash,
              &Result::continentalness_2A);
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[3].fork_hash,
              &Result::continentalness_3A);
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[4].fork_hash,
              &Result::continentalness_4A);
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[5].fork_hash,
              &Result::continentalness_5A);
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[6].fork_hash,
              &Result::continentalness_6A);
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[7].fork_hash,
              &Result::continentalness_7A);
  init_octave(noise_a_fork,
              device_chosen_continentalness_config.octaves_a[8].fork_hash,
              &Result::continentalness_8A);

  const auto noise_b_fork = noise_random.fork();
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[0].fork_hash,
              &Result::continentalness_0B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[1].fork_hash,
              &Result::continentalness_1B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[2].fork_hash,
              &Result::continentalness_2B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[3].fork_hash,
              &Result::continentalness_3B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[4].fork_hash,
              &Result::continentalness_4B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[5].fork_hash,
              &Result::continentalness_5B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[6].fork_hash,
              &Result::continentalness_6B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[7].fork_hash,
              &Result::continentalness_7B);
  init_octave(noise_b_fork,
              device_chosen_continentalness_config.octaves_b[8].fork_hash,
              &Result::continentalness_8B);
}
} // namespace KernelSeed1

struct SeedPos {
  uint32_t seed_index;
  int32_t x;
  int32_t z;
};

struct SeedPosACentre {
  float centre_val;
  uint32_t seed_index;
  int32_t x;
  int32_t z;
};

// ring a ring a rosie, bucket full of SeedPosARing
struct SeedPosARing {
  // float ring_val[16];
  uint64_t ring_val;
  float centre_val;
  uint32_t seed_index;
  int32_t x;
  int32_t z;

  __device__ float upperbound(uint32_t index) const {
    uint64_t bits = (ring_val >> (4 * index)) & 0xF;
    return (float)bits * (-0.062539f); // 1f / (-15.99f)
  }
};

template <typename T> __device__ T warp_reduce_add(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(-1u, val);
#else
  val += __shfl_down_sync(-1u, val, 1);
  val += __shfl_down_sync(-1u, val, 2);
  val += __shfl_down_sync(-1u, val, 4);
  val += __shfl_down_sync(-1u, val, 8);
  val += __shfl_down_sync(-1u, val, 16);
  return val;
#endif
}
__device__ float warp_reduce_max(float val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xFFFFFFFF, val);
#else
  float lane_val;
  lane_val = __shfl_down_sync(-1u, val, 1);
  val = fmaxf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 2);
  val = fmaxf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 4);
  val = fmaxf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 8);
  val = fmaxf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 16);
  val = fmaxf(lane_val, val);
  return val;
#endif
}
__device__ float warp_reduce_min(float val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(0xFFFFFFFF, val);
#else
  float lane_val;
  lane_val = __shfl_down_sync(-1u, val, 1);
  val = fminf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 2);
  val = fminf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 4);
  val = fminf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 8);
  val = fminf(lane_val, val);
  lane_val = __shfl_down_sync(-1u, val, 16);
  val = fminf(lane_val, val);
  return val;
#endif
}

namespace KernelFilterYOffset {
constexpr uint32_t threads_per_block = 256;
constexpr uint32_t threads_per_run = UINT64_C(1) << 19;
__device__ XrsrRandomFork noise_yo_fork(XrsrRandomFork noise_fork) {
  XrsrRandom rng{noise_fork.lo, noise_fork.hi};
  rng.nextInternal();
  return {rng.lo, rng.hi};
}

constexpr XrsrForkHash octave_yo_fork_hash(XrsrForkHash hash) {
  XrsrRandom rng{hash.lo, hash.hi};
  rng.nextInternal();
  return {rng.lo, rng.hi};
}

template <OctaveConfig octave_config>
__device__ float octave_yo_mod1(const XrsrRandomFork &noise_yo_fork) {
  constexpr auto fork_hash = octave_yo_fork_hash(octave_config.fork_hash);

  return ((uint32_t)noise_yo_fork.from(fork_hash).nextBits(32) & 0xFFFFFF) *
         5.9604645E-8f;
}

__global__ __launch_bounds__(threads_per_block) void kernel(
    uint64_t start_seed, OutputBuffer<uint64_t> outputs) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t seed = start_seed + index;
  const auto seed_fork = XrsrRandom(seed).fork();
  auto noise_random =
      seed_fork.from(device_chosen_continentalness_config.fork_hash);

  auto noise_a_yo_fork = noise_yo_fork(noise_random.fork());
  float c_0A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[0]>(
      noise_a_yo_fork);
  float c_1A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[1]>(
      noise_a_yo_fork);
  float c_2A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[2]>(
      noise_a_yo_fork);

  auto noise_b_yo_fork = noise_yo_fork(noise_random.fork());
  float c_0B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[0]>(
      noise_b_yo_fork);
  float c_1B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[1]>(
      noise_b_yo_fork);
  float c_2B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[2]>(
      noise_b_yo_fork);

  // TODO: test swapping 1A and 2A scores due to difference in yo plots with
  // COMMISSION
  /* COMMISSION score:
  float score = .35f * abs(c_0A_yo - .5f) + .35f * abs(c_0B_yo - .5f) +
                .11f * abs(c_1A_yo - .5f) + .11f * abs(c_1B_yo - .5f) +
                .04f * abs(c_2A_yo - .5f) + .04f * abs(c_2B_yo - .5f);
  */
  float score0A = abs(c_0A_yo - .5f);
  float score0B = abs(c_0B_yo - .5f);
  // float score2 = abs(c_2A_yo - .5f);
  if (score0A >= 0.15 || score0B >= 0.2) // || score2 >= 0.25f)
    return;                              // 1 in 2700

  uint32_t result_index = atomicAdd(&outputs.len, 1);
  if (result_index >= outputs.max_len)
    return;
  outputs.data[result_index] = seed;
}

void run(uint64_t start_seed, OutputBuffer<uint64_t> outputs) {
  kernel<<<threads_per_run / threads_per_block, threads_per_block>>>(start_seed,
                                                                     outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilterYOffset
namespace KernelFilter0A_Coarse {
constexpr uint32_t threads_per_block = 256;
constexpr uint32_t threads_per_seed_sqrt = UINT64_C(1) << 8;
constexpr uint32_t threads_per_seed =
    threads_per_seed_sqrt * threads_per_seed_sqrt;
constexpr int32_t pos_range =
    524288 / 4; // 0A tiling : world coords -> biome coords
constexpr int32_t pos_step = pos_range / threads_per_seed_sqrt;
static_assert(pos_step >= 1);
static_assert(pos_range <= 60'000'000 / 4);

__global__ __launch_bounds__(threads_per_block) void kernel(
    InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs) {
  // uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t seeds_len = seeds.len;
  // uint64_t seed;

  __shared__ GradDotTable shared_grad_dot_table;
  if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[threadIdx.x] =
        reinterpret_cast<uint32_t *>(&device_grad_dot_table)[threadIdx.x];
  }
  for (uint32_t seed_index = blockIdx.x; seed_index < seeds_len;
       seed_index += gridDim.x) {

    __syncthreads();
    // seed = seeds.data[seed_index];
    __shared__ KernelSeed1::ResultSampler<1> shared_octaves;
    if (threadIdx.x < sizeof(shared_octaves) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&shared_octaves)[threadIdx.x] =
          reinterpret_cast<uint32_t *>(
              &KernelSeed1::results[seed_index])[threadIdx.x];
    }
    __syncthreads();

    for (uint32_t pos_index = threadIdx.x; pos_index < threads_per_seed;
         pos_index += blockDim.x) {
      uint32_t x_index = pos_index % threads_per_seed_sqrt;
      uint32_t z_index = pos_index / threads_per_seed_sqrt;

      int32_t x = (int32_t)x_index * pos_step - pos_range / 2;
      int32_t z = (int32_t)z_index * pos_step - pos_range / 2;

      float val = shared_octaves.sample(shared_grad_dot_table, x, 0, z);
      if (val >= PREFILTER_THRESHOLD_1)
        continue;

      uint32_t result_index = atomicAdd(&outputs.len, 1);
      if (result_index >= outputs.max_len)
        return;
      outputs.data[result_index] = {seed_index, x, z};
    }
  }
}

void run(InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs) {
  kernel<<<2048, threads_per_block>>>(seeds, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter0A_Coarse

namespace KernelFilter0A_Medium {

constexpr uint32_t threads_per_block = 256;
constexpr uint32_t threads_per_seed_w = 8;
constexpr uint32_t threads_per_seed_h = 4;
constexpr uint32_t threads_per_seed = threads_per_seed_w * threads_per_seed_h;

constexpr int32_t pos_range = SMALL_PREFILTER_RADIUS / 4;
constexpr int32_t pos_x_step = pos_range / (threads_per_seed_w - 1);
constexpr int32_t pos_z_step = pos_range / (threads_per_seed_h - 1);

__global__ __launch_bounds__(threads_per_block) void kernel(
    InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {

  uint32_t total_threads = inputs.len * threads_per_seed;

  __shared__ GradDotTable shared_grad_dot_table;
  if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[threadIdx.x] =
        reinterpret_cast<uint32_t *>(&device_grad_dot_table)[threadIdx.x];
  }
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total_threads; index += gridDim.x * blockDim.x) {
    __syncthreads();
    uint32_t input_index = index / threads_per_seed;
    uint32_t pos_index = index % threads_per_seed;

    uint32_t warp_index = threadIdx.x / threads_per_seed;

    const auto input = inputs.data[input_index];
    __shared__ KernelSeed1::ResultSampler<1> shared_octaves[8];
    for (uint32_t i = pos_index;
         i < sizeof(shared_octaves[0]) / sizeof(uint32_t);
         i += threads_per_seed) {
      reinterpret_cast<uint32_t *>(&shared_octaves[warp_index])[i] =
          reinterpret_cast<uint32_t *>(
              &KernelSeed1::results[input.seed_index])[i];
    }
    __syncthreads();

    uint32_t x_index = pos_index % threads_per_seed_w;
    uint32_t z_index = pos_index / threads_per_seed_w;

    int32_t x = (int32_t)x_index * pos_x_step - pos_range / 2;
    int32_t z = (int32_t)z_index * pos_z_step - pos_range / 2;

    float val = shared_octaves[warp_index].sample(shared_grad_dot_table,
                                                  input.x + x, 0, input.z + z);

    float min_val = warp_reduce_min(val);
    min_val = __shfl_sync(-1u, min_val, 0);
    unsigned mask = __ballot_sync(-1u, min_val == val);
    int min_lane = __ffs(mask) - 1;

    if (pos_index == min_lane) {
      if (min_val >= PREFILTER_THRESHOLD_2)
        continue;

      uint32_t result_index = atomicAdd(&outputs.len, 1);
      if (result_index >= outputs.max_len)
        return;
      outputs.data[result_index] = {input.seed_index, input.x + x, input.z + z};
    }
  }
}

void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
  kernel<<<32 * 256, threads_per_block>>>(inputs, outputs);
  TRY_CUDA(cudaGetLastError());
}

} // namespace KernelFilter0A_Medium

namespace KernelFilter3B_Centre {
constexpr uint32_t threads_per_block = 256;
constexpr uint32_t blocks = 4096;
constexpr uint32_t threads_per_seed_sqrt = 114; // max 113
constexpr uint32_t threads_per_seed =
    threads_per_seed_sqrt * threads_per_seed_sqrt;
constexpr int32_t pos_step = 524288 / 4;
constexpr int32_t pos_range = (int32_t)threads_per_seed_sqrt * pos_step;
static_assert(pos_step >= 1);
static_assert(pos_range <= 60'000'000 / 4);

__global__ __launch_bounds__(threads_per_block) void kernel(
    InputBuffer<SeedPosARing> inputs, OutputBuffer<SeedPosARing> outputs) {

  __shared__ GradDotTable shared_grad_dot_table;
  for (uint32_t i = threadIdx.x;
       i < sizeof(shared_grad_dot_table) / sizeof(uint32_t); i += blockDim.x) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[i] =
        reinterpret_cast<const uint32_t *>(&device_grad_dot_table)[i];
  }

  for (uint32_t input_index = blockIdx.x; input_index < inputs.len;
       input_index += gridDim.x) {
    __syncthreads();
    const auto input = inputs.data[input_index];

    __shared__ KernelSeed1::ResultSampler<8> shared_octaves;
    for (uint32_t i = threadIdx.x;
         i < sizeof(shared_octaves) / sizeof(uint32_t); i += blockDim.x) {
      reinterpret_cast<uint32_t *>(&shared_octaves)[i] =
          reinterpret_cast<uint32_t *>(
              &KernelSeed1::results[input.seed_index])[i];
    }

    __syncthreads();

    for (uint32_t pos_index = threadIdx.x; pos_index < threads_per_seed;
         pos_index += blockDim.x) {
      uint32_t x_index = pos_index % threads_per_seed_sqrt;
      uint32_t z_index = pos_index / threads_per_seed_sqrt;

      int32_t dx = (int32_t)x_index * pos_step - pos_range / 2;
      int32_t dz = (int32_t)z_index * pos_step - pos_range / 2;

      float val_0b = sample_octave<chosen_continentalness_config.octaves_b[0]>(
          shared_grad_dot_table, shared_octaves.octaves[1], input.x + dx, 0,
          input.z + dz);

      // if (val_0b > -0.4f)
      //  continue;

      float val_1b = sample_octave<chosen_continentalness_config.octaves_b[1]>(
          shared_grad_dot_table, shared_octaves.octaves[3], input.x + dx, 0,
          input.z + dz);

      float val_2b = sample_octave<chosen_continentalness_config.octaves_b[2]>(
          shared_grad_dot_table, shared_octaves.octaves[5], input.x + dx, 0,
          input.z + dz);

      float val_3b = sample_octave<chosen_continentalness_config.octaves_b[3]>(
          shared_grad_dot_table, shared_octaves.octaves[7], input.x + dx, 0,
          input.z + dz);

      float val = (val_0b + val_1b + val_2b + val_3b);

      if (val + input.centre_val < CENTRE_THRESHOLD_2)
        continue;

      uint32_t result_index = atomicAdd(&outputs.len, 1);
      if (result_index >= outputs.max_len)
        continue;
      outputs.data[result_index] = {input.ring_val, input.centre_val,
                                    input.seed_index, input.x + dx,
                                    input.z + dz};
    }

    __syncthreads();
  }
}

void run(InputBuffer<SeedPosARing> inputs, OutputBuffer<SeedPosARing> outputs) {
  kernel<<<blocks, threads_per_block>>>(inputs, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter3B_Centre

namespace KernelFilter3A_Centre {

constexpr uint32_t threads_per_block = 256;
constexpr uint32_t threads_per_seed_w = 8;
constexpr uint32_t threads_per_seed_h = 4;
constexpr uint32_t threads_per_seed = threads_per_seed_w * threads_per_seed_h;

constexpr int32_t pos_range = FINAL_PREFILTER_RADIUS / 4;
constexpr int32_t pos_x_step = pos_range / (threads_per_seed_w - 1);
constexpr int32_t pos_z_step = pos_range / (threads_per_seed_h - 1);

__global__ __launch_bounds__(threads_per_block) void kernel(
    InputBuffer<SeedPos> inputs, OutputBuffer<SeedPosACentre> outputs) {

  uint32_t total_threads = inputs.len * threads_per_seed;

  __shared__ GradDotTable shared_grad_dot_table;
  if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[threadIdx.x] =
        reinterpret_cast<uint32_t *>(&device_grad_dot_table)[threadIdx.x];
  }
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total_threads; index += gridDim.x * blockDim.x) {
    __syncthreads();
    uint32_t input_index = index / threads_per_seed;
    uint32_t pos_index = index % threads_per_seed;

    uint32_t warp_index = threadIdx.x / threads_per_seed;

    const auto input = inputs.data[input_index];
    __shared__ KernelSeed1::ResultSampler<7> shared_octaves[8];
    for (uint32_t i = pos_index;
         i < sizeof(shared_octaves[0]) / sizeof(uint32_t);
         i += threads_per_seed) {
      reinterpret_cast<uint32_t *>(&shared_octaves[warp_index])[i] =
          reinterpret_cast<uint32_t *>(
              &KernelSeed1::results[input.seed_index])[i];
    }
    __syncthreads();

    uint32_t x_index = pos_index % threads_per_seed_w;
    uint32_t z_index = pos_index / threads_per_seed_w;

    int32_t x = (int32_t)x_index * pos_x_step - pos_range / 2;
    int32_t z = (int32_t)z_index * pos_z_step - pos_range / 2;

    float val_0a = sample_octave<chosen_continentalness_config.octaves_a[0]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[0],
        input.x + x, 0, input.z + z);

    float val_1a = sample_octave<chosen_continentalness_config.octaves_a[1]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[2],
        input.x + x, 0, input.z + z);

    float val_2a = sample_octave<chosen_continentalness_config.octaves_a[2]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[4],
        input.x + x, 0, input.z + z);

    float val_3a = sample_octave<chosen_continentalness_config.octaves_a[3]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[6],
        input.x + x, 0, input.z + z);

    float val = val_0a + val_1a + val_2a + val_3a;

    float max_val = warp_reduce_max(val);
    max_val = __shfl_sync(-1u, max_val, 0);
    unsigned mask = __ballot_sync(-1u, max_val == val);
    int max_lane = __ffs(mask) - 1;

    if (pos_index == max_lane) {
      if (max_val <= CENTRE_THRESHOLD_1)
        continue;

      uint32_t result_index = atomicAdd(&outputs.len, 1);
      if (result_index >= outputs.max_len)
        continue;
      outputs.data[result_index] = {max_val, input.seed_index, input.x + x,
                                    input.z + z};
    }
  }
}

void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPosACentre> outputs) {
  kernel<<<32 * 256, threads_per_block>>>(inputs, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter3A_Centre

namespace KernelFilter3A_Rings {

constexpr uint32_t ring_radius = RING_RADIUS / 4;
__device__ __constant__ constexpr int32_t ring_pos[16][2] = {
    {(int32_t)(0.0 * ring_radius), (int32_t)(1.0 * ring_radius)},
    {(int32_t)(0.38 * ring_radius), (int32_t)(0.92 * ring_radius)},
    {(int32_t)(0.71 * ring_radius), (int32_t)(0.71 * ring_radius)},
    {(int32_t)(0.92 * ring_radius), (int32_t)(0.38 * ring_radius)},
    {(int32_t)(1.0 * ring_radius), (int32_t)(0.0 * ring_radius)},
    {(int32_t)(0.92 * ring_radius), (int32_t)(-0.38 * ring_radius)},
    {(int32_t)(0.71 * ring_radius), (int32_t)(-0.71 * ring_radius)},
    {(int32_t)(0.38 * ring_radius), (int32_t)(-0.92 * ring_radius)},
    {(int32_t)(0.0 * ring_radius), (int32_t)(-1.0 * ring_radius)},
    {(int32_t)(-0.38 * ring_radius), (int32_t)(-0.92 * ring_radius)},
    {(int32_t)(-0.71 * ring_radius), (int32_t)(-0.71 * ring_radius)},
    {(int32_t)(-0.92 * ring_radius), (int32_t)(-0.38 * ring_radius)},
    {(int32_t)(-1.0 * ring_radius), (int32_t)(-0.0 * ring_radius)},
    {(int32_t)(-0.92 * ring_radius), (int32_t)(0.38 * ring_radius)},
    {(int32_t)(-0.71 * ring_radius), (int32_t)(0.71 * ring_radius)},
    {(int32_t)(-0.38 * ring_radius), (int32_t)(0.92 * ring_radius)}};

constexpr uint32_t threads_per_block = 256;

constexpr uint32_t threads_per_seed = 16;
static_assert(threads_per_seed == 16);

__global__ __launch_bounds__(threads_per_block) void kernel(
    InputBuffer<SeedPosACentre> inputs, OutputBuffer<SeedPosARing> outputs) {

  uint32_t total_threads = inputs.len * threads_per_seed;

  __shared__ GradDotTable shared_grad_dot_table;
  if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[threadIdx.x] =
        reinterpret_cast<uint32_t *>(&device_grad_dot_table)[threadIdx.x];
  }
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total_threads; index += gridDim.x * blockDim.x) {

    __syncthreads();
    uint32_t input_index = index / threads_per_seed;
    uint32_t pos_index = index % threads_per_seed;
    uint32_t warp_index = threadIdx.x / 16;

    const auto input = inputs.data[input_index];
    __shared__ KernelSeed1::ResultSampler<7> shared_octaves[16];
    for (uint32_t i = pos_index;
         i < sizeof(shared_octaves[0]) / sizeof(uint32_t);
         i += threads_per_seed) {
      reinterpret_cast<uint32_t *>(&shared_octaves[warp_index])[i] =
          reinterpret_cast<uint32_t *>(
              &KernelSeed1::results[input.seed_index])[i];
    }
    __syncthreads();

    int32_t dx = ring_pos[pos_index][0];
    int32_t dz = ring_pos[pos_index][1];

    float val_0a = sample_octave<chosen_continentalness_config.octaves_a[0]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[0],
        input.x + dx, 0, input.z + dz);

    float val_1a = sample_octave<chosen_continentalness_config.octaves_a[1]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[2],
        input.x + dx, 0, input.z + dz);

    float val_2a = sample_octave<chosen_continentalness_config.octaves_a[2]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[4],
        input.x + dx, 0, input.z + dz);

    float val_3a = sample_octave<chosen_continentalness_config.octaves_a[3]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[6],
        input.x + dx, 0, input.z + dz);

    float val = (val_0a + val_1a + val_2a + val_3a);
    float centre_diff = val - input.centre_val;

    centre_diff += __shfl_down_sync(-1u, centre_diff, 1);
    centre_diff += __shfl_down_sync(-1u, centre_diff, 2);
    centre_diff += __shfl_down_sync(-1u, centre_diff, 4);
    centre_diff += __shfl_down_sync(-1u, centre_diff, 8);

    uint64_t ring_val = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      float temp = __shfl_sync(-1u, val, i, 16);
      // if (pos_index == 0)
      uint64_t fourbits = (uint64_t)(15.99f * __saturatef(-val));
      ring_val |= fourbits << (i * 4);
    }

    // valid results in lanes 0 and 16 (pos_index==0)
    if (pos_index == 0) {

      if (centre_diff > SCORE_REQUIREMENT)
        continue;

      uint32_t result_index = atomicAdd(&outputs.len, 1);
      if (result_index >= outputs.max_len)
        continue;
      outputs.data[result_index] = {ring_val, input.centre_val,
                                    input.seed_index, input.x, input.z};
    }
  }
}

void run(InputBuffer<SeedPosACentre> inputs,
         OutputBuffer<SeedPosARing> outputs) {
  kernel<<<32 * 256, threads_per_block>>>(inputs, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter3A_Rings

namespace KernelFilter3B_Rings {

constexpr uint32_t ring_radius = RING_RADIUS / 4;
__device__ __constant__ constexpr int32_t ring_pos[16][2] = {
    {(int32_t)(0.0 * ring_radius), (int32_t)(1.0 * ring_radius)},
    {(int32_t)(0.38 * ring_radius), (int32_t)(0.92 * ring_radius)},
    {(int32_t)(0.71 * ring_radius), (int32_t)(0.71 * ring_radius)},
    {(int32_t)(0.92 * ring_radius), (int32_t)(0.38 * ring_radius)},
    {(int32_t)(1.0 * ring_radius), (int32_t)(0.0 * ring_radius)},
    {(int32_t)(0.92 * ring_radius), (int32_t)(-0.38 * ring_radius)},
    {(int32_t)(0.71 * ring_radius), (int32_t)(-0.71 * ring_radius)},
    {(int32_t)(0.38 * ring_radius), (int32_t)(-0.92 * ring_radius)},
    {(int32_t)(0.0 * ring_radius), (int32_t)(-1.0 * ring_radius)},
    {(int32_t)(-0.38 * ring_radius), (int32_t)(-0.92 * ring_radius)},
    {(int32_t)(-0.71 * ring_radius), (int32_t)(-0.71 * ring_radius)},
    {(int32_t)(-0.92 * ring_radius), (int32_t)(-0.38 * ring_radius)},
    {(int32_t)(-1.0 * ring_radius), (int32_t)(-0.0 * ring_radius)},
    {(int32_t)(-0.92 * ring_radius), (int32_t)(0.38 * ring_radius)},
    {(int32_t)(-0.71 * ring_radius), (int32_t)(0.71 * ring_radius)},
    {(int32_t)(-0.38 * ring_radius), (int32_t)(0.92 * ring_radius)}};

constexpr uint32_t threads_per_block = 256;

constexpr uint32_t threads_per_seed = 16;
static_assert(threads_per_seed == 16);

__global__ __launch_bounds__(threads_per_block) void kernel(
    InputBuffer<SeedPosARing> inputs, OutputBuffer<SeedPos> outputs) {

  uint32_t total_threads = inputs.len * threads_per_seed;

  __shared__ GradDotTable shared_grad_dot_table;
  if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[threadIdx.x] =
        reinterpret_cast<uint32_t *>(&device_grad_dot_table)[threadIdx.x];
  }
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total_threads; index += gridDim.x * blockDim.x) {

    __syncthreads();
    uint32_t input_index = index / threads_per_seed;
    uint32_t pos_index = index % threads_per_seed;
    uint32_t warp_index = threadIdx.x / 16;

    const auto input = inputs.data[input_index];
    __shared__ KernelSeed1::ResultSampler<8> shared_octaves[16];
    for (uint32_t i = pos_index;
         i < sizeof(shared_octaves[0]) / sizeof(uint32_t);
         i += threads_per_seed) {
      reinterpret_cast<uint32_t *>(&shared_octaves[warp_index])[i] =
          reinterpret_cast<uint32_t *>(
              &KernelSeed1::results[input.seed_index])[i];
    }
    __syncthreads();

    int32_t dx = ring_pos[pos_index][0];
    int32_t dz = ring_pos[pos_index][1];

    float val_0b = sample_octave<chosen_continentalness_config.octaves_b[0]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[1],
        input.x + dx, 0, input.z + dz);

    float val_1b = sample_octave<chosen_continentalness_config.octaves_b[1]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[3],
        input.x + dx, 0, input.z + dz);

    float val_2b = sample_octave<chosen_continentalness_config.octaves_b[2]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[5],
        input.x + dx, 0, input.z + dz);

    float val_3b = sample_octave<chosen_continentalness_config.octaves_b[3]>(
        shared_grad_dot_table, shared_octaves[warp_index].octaves[7],
        input.x + dx, 0, input.z + dz);

    float val = (val_0b + val_1b + val_2b + val_3b);
    float valAB = val + input.upperbound(pos_index); //.ring_val[pos_index];

    int32_t passed = (int32_t)(valAB < RING_POINT_THRESHOLD);

    passed += __shfl_down_sync(-1u, passed, 1);
    passed += __shfl_down_sync(-1u, passed, 2);
    passed += __shfl_down_sync(-1u, passed, 4);
    passed += __shfl_down_sync(-1u, passed, 8);

    // valid results in lanes 0 and 16 (pos_index==0)
    if (pos_index == 0) {

      if (passed < RING_POINT_MIN_PASSED)
        continue;

      uint32_t result_index = atomicAdd(&outputs.len, 1);
      if (result_index >= outputs.max_len)
        continue;
      outputs.data[result_index] = {input.seed_index, input.x, input.z};
    }
  }
}

void run(InputBuffer<SeedPosARing> inputs, OutputBuffer<SeedPos> outputs) {
  kernel<<<32 * 256, threads_per_block>>>(inputs, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter3B_Rings

namespace KernelFilterIsland {

constexpr uint32_t threads_per_block = 256;
constexpr uint32_t threads_per_seed_w = 8;
constexpr uint32_t threads_per_seed_h = 4;
constexpr uint32_t threads_per_seed = threads_per_seed_w * threads_per_seed_h;

constexpr int32_t pos_range = ISLAND_SEARCH_RADIUS / 4;
constexpr int32_t pos_x_step = pos_range / (threads_per_seed_w - 1);
constexpr int32_t pos_z_step = pos_range / (threads_per_seed_h - 1);

__global__ __launch_bounds__(threads_per_block) void kernel(
    InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {

  uint32_t total_threads = inputs.len * threads_per_seed;

  __shared__ GradDotTable shared_grad_dot_table;
  if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[threadIdx.x] =
        reinterpret_cast<uint32_t *>(&device_grad_dot_table)[threadIdx.x];
  }
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total_threads; index += gridDim.x * blockDim.x) {
    __syncthreads();
    uint32_t input_index = index / threads_per_seed;
    uint32_t pos_index = index % threads_per_seed;

    uint32_t warp_index = threadIdx.x / threads_per_seed;

    const auto input = inputs.data[input_index];
    __shared__ KernelSeed1::ResultSampler<18> shared_octaves[8];
    for (uint32_t i = pos_index;
         i < sizeof(shared_octaves[0]) / sizeof(uint32_t);
         i += threads_per_seed) {
      reinterpret_cast<uint32_t *>(&shared_octaves[warp_index])[i] =
          reinterpret_cast<uint32_t *>(
              &KernelSeed1::results[input.seed_index])[i];
    }
    __syncthreads();

    uint32_t x_index = pos_index % threads_per_seed_w;
    uint32_t z_index = pos_index / threads_per_seed_w;

    int32_t x = (int32_t)x_index * pos_x_step - pos_range / 2;
    int32_t z = (int32_t)z_index * pos_z_step - pos_range / 2;

    float val = shared_octaves[warp_index].sample(shared_grad_dot_table,
                                                  input.x + x, 0, input.z + z);

    float max_val = warp_reduce_max(val);
    max_val = __shfl_sync(-1u, max_val, 0);
    unsigned mask = __ballot_sync(-1u, max_val == val);
    int max_lane = __ffs(mask) - 1;

    if (pos_index == max_lane) {
      if (max_val <= ISLAND_THRESHOLD)
        continue;

      uint32_t result_index = atomicAdd(&outputs.len, 1);
      if (result_index >= outputs.max_len)
        continue;
      outputs.data[result_index] = {input.seed_index, input.x + x, input.z + z};
    }
  }
}

void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
  kernel<<<256, threads_per_block>>>(inputs, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilterIsland
constexpr bool is_pow2(uint32_t val) { return (val & (val - 1)) == 0; }

constexpr uint32_t log2(uint32_t val) { return 31 - std::countl_zero(val); }

struct CudaEventWrapper {
  cudaEvent_t event;

  CudaEventWrapper() : event(nullptr) { TRY_CUDA(cudaEventCreate(&event)); }

  CudaEventWrapper(CudaEventWrapper &&other) : event(other.event) {
    other.event = nullptr;
  }

  ~CudaEventWrapper() {
    if (event == nullptr)
      return;
    TRY_CUDA(cudaEventDestroy(event));
  }

  void record(cudaStream_t stream = 0) const {
    TRY_CUDA(cudaEventRecord(event, stream));
  }

  float elapsed(const CudaEventWrapper &end) const {
    float ms;
    TRY_CUDA(cudaEventElapsedTime(&ms, event, end.event));
    return ms;
  }

  void synchronize() const { TRY_CUDA(cudaEventSynchronize(event)); }
};

struct BufferLens {
  uint32_t results_len_filter_0A_coarse;
  uint32_t results_len_filter_0A_med;
  uint32_t results_len_filter_3AC;
  uint32_t results_len_filter_3AR;
  uint32_t results_len_filter_3BC;
  uint32_t results_len_filter_3BR;
  uint32_t results_len_filter_island;
  uint32_t results_len_filter_yoffset;
};

uint64_t random_start_seed() {
  std::random_device device;
  return ((uint64_t)device() << 32) + (uint64_t)device();
}

GpuThread::GpuThread(int device, GpuOutputs &outputs)
    : Thread(), device(device), outputs(outputs) {
  start();
}

void GpuThread::run() {
  std::printf("Initializing device %d\n", device);

  TRY_CUDA(cudaSetDevice(device));
  init_grad_dot_table();

  BufferLens host_buffer_lens;
  BufferLens *device_buffer_lens;
  TRY_CUDA(cudaMalloc(&device_buffer_lens, sizeof(*device_buffer_lens)));

  uint64_t start_seed = random_start_seed();
  // uint64_t start_seed = 9849470875906027758;
  std::printf("Running device %d at %" PRIu64 "\n", device, start_seed);

  DeviceBuffer buffer_seeds(sizeof(uint64_t) * KernelSeed1::threads_per_run);
  DeviceBuffer buffer_1(UINT32_C(1) << 31);
  DeviceBuffer buffer_2(UINT32_C(1) << 29);
  std::vector<SeedPos> h_buffer;

  namespace Filter0ACoarse = KernelFilter0A_Coarse;
  namespace Filter0AMed = KernelFilter0A_Medium;
  namespace Filter3AC = KernelFilter3A_Centre;
  namespace Filter3AR = KernelFilter3A_Rings;
  namespace Filter3BC = KernelFilter3B_Centre;
  namespace Filter3BR = KernelFilter3B_Rings;
  namespace FilterIsland = KernelFilterIsland;

  OutputBuffer<uint64_t> outputs_filter_yoffset(
      buffer_seeds, device_buffer_lens->results_len_filter_yoffset);
  uint32_t &host_outputs_filter_yoffset_len =
      host_buffer_lens.results_len_filter_yoffset;

  OutputBuffer<SeedPos> outputs_filter_0A_coarse(
      buffer_1, device_buffer_lens->results_len_filter_0A_coarse);
  uint32_t &host_outputs_filter_0A_coarse_len =
      host_buffer_lens.results_len_filter_0A_coarse;

  OutputBuffer<SeedPos> outputs_filter_0A_med(
      buffer_2, device_buffer_lens->results_len_filter_0A_med);
  uint32_t &host_outputs_filter_0A_med_len =
      host_buffer_lens.results_len_filter_0A_med;

  OutputBuffer<SeedPosACentre> outputs_filter_3AC(
      buffer_1, device_buffer_lens->results_len_filter_3AC);
  uint32_t &host_outputs_filter_3AC_len =
      host_buffer_lens.results_len_filter_3AC;

  OutputBuffer<SeedPosARing> outputs_filter_3AR(
      buffer_2, device_buffer_lens->results_len_filter_3AR);
  uint32_t &host_outputs_filter_3AR_len =
      host_buffer_lens.results_len_filter_3AR;

  OutputBuffer<SeedPosARing> outputs_filter_3BC(
      buffer_1, device_buffer_lens->results_len_filter_3BC);
  uint32_t &host_outputs_filter_3BC_len =
      host_buffer_lens.results_len_filter_3BC;

  OutputBuffer<SeedPos> outputs_filter_3BR(
      buffer_2, device_buffer_lens->results_len_filter_3BR);
  uint32_t &host_outputs_filter_3BR_len =
      host_buffer_lens.results_len_filter_3BR;

  OutputBuffer<SeedPos> outputs_filter_island(
      buffer_1, device_buffer_lens->results_len_filter_island);
  uint32_t &host_outputs_filter_island_len =
      host_buffer_lens.results_len_filter_island;

  CudaEventWrapper event_start, event_yoffset, event_seed_1,
      event_filter_0A_coarse, event_filter_0A_med, event_filter_3AC,
      event_filter_3AR, event_filter_3BC, event_filter_3BR, event_filter_island;

  int print_interval = 1;
  double time_yoffset = 0.0;
  double time_seed_1 = 0.0;
  double time_filter_0A_coarse = 0.0;
  double time_filter_0A_med = 0.0;
  double time_filter_3BC = 0.0;
  double time_filter_3AC = 0.0;
  double time_filter_3AR = 0.0;
  double time_filter_3BR = 0.0;
  double time_filter_island = 0.0;
  uint64_t inputs_yoffset = 0;
  uint64_t total_outputs_len_filter_yoffset = 0;
  uint64_t inputs_seed_1 = 0;
  uint64_t inputs_filter_0A_coarse = 0;
  uint64_t total_outputs_len_filter_0A_coarse = 0;
  uint64_t total_outputs_len_filter_0A_med = 0;
  uint64_t total_outputs_len_filter_3BC = 0;
  uint64_t total_outputs_len_filter_3AC = 0;
  uint64_t total_outputs_len_filter_3AR = 0;
  uint64_t total_outputs_len_filter_3BR = 0;
  uint64_t total_outputs_len_filter_island = 0;

  auto start = std::chrono::steady_clock::now();

  uint64_t currently_used_start_seed = 0;
  uint32_t start_seed_index = UINT32_MAX;

  for (uint32_t i = 0; !should_stop(); i++) {
    TRY_CUDA(
        cudaMemsetAsync(device_buffer_lens, 0, sizeof(*device_buffer_lens)));

    event_start.record();

    KernelFilterYOffset::run(start_seed, outputs_filter_yoffset);
    start_seed += KernelFilterYOffset::threads_per_run;
    TRY_CUDA(cudaGetLastError());

    event_yoffset.record();
    {
      KernelSeed1::kernel<<<KernelSeed1::threads_per_run /
                                KernelSeed1::threads_per_block,
                            KernelSeed1::threads_per_block>>>(
          outputs_filter_yoffset);
    }
    event_seed_1.record();

    {
      Filter0ACoarse::run(outputs_filter_yoffset, outputs_filter_0A_coarse);
    }
    event_filter_0A_coarse.record();
    {
      Filter0AMed::run(outputs_filter_0A_coarse, outputs_filter_0A_med);
    }
    event_filter_0A_med.record();
    {
      Filter3AC::run(outputs_filter_0A_med, outputs_filter_3AC);
    }
    event_filter_3AC.record();

    {
      Filter3AR::run(outputs_filter_3AC, outputs_filter_3AR);
    }
    event_filter_3AR.record();

    {
      Filter3BC::run(outputs_filter_3AR, outputs_filter_3BC);
    }
    event_filter_3BC.record();

    {
      Filter3BR::run(outputs_filter_3BC, outputs_filter_3BR);
    }
    event_filter_3BR.record();
    {
      FilterIsland::run(outputs_filter_3BR, outputs_filter_island);
    }
    event_filter_island.record();

    TRY_CUDA(cudaMemcpyAsync(&host_buffer_lens, device_buffer_lens,
                             sizeof(host_buffer_lens), cudaMemcpyDeviceToHost));

    event_filter_island.synchronize();

    time_yoffset += event_start.elapsed(event_yoffset) * 1e-3;
    time_seed_1 += event_yoffset.elapsed(event_seed_1) * 1e-3;
    time_filter_0A_coarse +=
        event_seed_1.elapsed(event_filter_0A_coarse) * 1e-3;
    time_filter_0A_med +=
        event_filter_0A_coarse.elapsed(event_filter_0A_med) * 1e-3;
    time_filter_3AC += event_filter_0A_med.elapsed(event_filter_3AC) * 1e-3;
    time_filter_3AR += event_filter_3AC.elapsed(event_filter_3AR) * 1e-3;
    time_filter_3BC += event_filter_3AR.elapsed(event_filter_3BC) * 1e-3;
    time_filter_3BR += event_filter_3BC.elapsed(event_filter_3BR) * 1e-3;
    time_filter_island += event_filter_3BR.elapsed(event_filter_island) * 1e-3;

    total_outputs_len_filter_yoffset += host_outputs_filter_yoffset_len;
    total_outputs_len_filter_0A_coarse += host_outputs_filter_0A_coarse_len;
    total_outputs_len_filter_0A_med += host_outputs_filter_0A_med_len;
    total_outputs_len_filter_3AC += host_outputs_filter_3AC_len;
    total_outputs_len_filter_3AR += host_outputs_filter_3AR_len;
    total_outputs_len_filter_3BC += host_outputs_filter_3BC_len;
    total_outputs_len_filter_3BR += host_outputs_filter_3BR_len;
    total_outputs_len_filter_island += host_outputs_filter_island_len;

    if (host_outputs_filter_yoffset_len > outputs_filter_yoffset.max_len) {
      std::printf("outputs_yoffset.len > outputs_yoffset.max_len : %" PRIu32
                  " > %" PRIu32 "\n",
                  host_outputs_filter_yoffset_len,
                  outputs_filter_yoffset.max_len);
    }
    if (host_outputs_filter_0A_coarse_len > outputs_filter_0A_coarse.max_len) {
      std::printf("outputs_filter_1.len > outputs_filter_1.max_len : %" PRIu32
                  " > %" PRIu32 "\n",
                  host_outputs_filter_0A_coarse_len,
                  outputs_filter_0A_coarse.max_len);
    }
    if (host_outputs_filter_0A_med_len > outputs_filter_0A_med.max_len) {
      std::printf("outputs_filter_1.len > outputs_filter_1.max_len : %" PRIu32
                  " > %" PRIu32 "\n",
                  host_outputs_filter_0A_med_len,
                  outputs_filter_0A_med.max_len);
    }
    if (host_outputs_filter_3AC_len > outputs_filter_3AC.max_len) {
      std::printf(
          "outputs_filter_3AC.len > outputs_filter_3A.max_len : %" PRIu32
          " > %" PRIu32 "\n",
          host_outputs_filter_3AC_len, outputs_filter_3AC.max_len);
    }
    if (host_outputs_filter_3AR_len > outputs_filter_3AR.max_len) {
      std::printf(
          "outputs_filter_3AR.len > outputs_filter_3A.max_len : %" PRIu32
          " > %" PRIu32 "\n",
          host_outputs_filter_3AR_len, outputs_filter_3AR.max_len);
    }
    if (host_outputs_filter_3BC_len > outputs_filter_3BC.max_len) {
      std::printf(
          "outputs_filter_3BC.len > outputs_filter_3BC.max_len : %" PRIu32
          " > %" PRIu32 "\n",
          host_outputs_filter_3BC_len, outputs_filter_3BC.max_len);
    }
    if (host_outputs_filter_3BR_len > outputs_filter_3BR.max_len) {
      std::printf(
          "outputs_filter_3BR.len > outputs_filter_3BR.max_len : %" PRIu32
          " > %" PRIu32 "\n",
          host_outputs_filter_3BR_len, outputs_filter_3BR.max_len);
    }
    if (host_outputs_filter_island_len > outputs_filter_island.max_len) {
      std::printf(
          "outputs_filter_island.len > outputs_filter_island.max_len : %" PRIu32
          " > %" PRIu32 "\n",
          host_outputs_filter_island_len, outputs_filter_island.max_len);
    }

    const auto &final_outputs = outputs_filter_island;
    const auto &final_outputs_len = host_outputs_filter_island_len;
    if (final_outputs_len > 0) {
      uint32_t len = final_outputs_len;
      h_buffer.resize(len);
      TRY_CUDA(cudaMemcpy(h_buffer.data(), final_outputs.data,
                          sizeof(*h_buffer.data()) * len,
                          cudaMemcpyDeviceToHost));
      {
        std::lock_guard lock(outputs.mutex);
        for (const auto &result : h_buffer) {
          uint64_t seed;
          TRY_CUDA(cudaMemcpy(&seed,
                              &outputs_filter_yoffset.data[result.seed_index],
                              sizeof(seed), cudaMemcpyDeviceToHost));
          outputs.queue.push({seed, result.x * 4, result.z * 4});
        }
      }
    }

    if ((i + 1) % print_interval == 0) {
      auto end = std::chrono::steady_clock::now();
      double time_total =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() *
          1e-9;

      double kernel_time_total =
          time_yoffset + time_seed_1 + time_filter_0A_coarse +
          time_filter_0A_med + time_filter_3BC + time_filter_3AC +
          time_filter_3AR + time_filter_3BR + time_filter_island;
      std::printf("\n");
      std::printf("f_yoffset   - %9.3f ms | %6.3f %% | %12d"
                  " -> %12" PRIu64 " | 1 in %11.3f"
                  " | %7.3f Gsps\n",
                  time_yoffset * 1e3, time_yoffset / time_total * 100.0,
                  print_interval * KernelFilterYOffset::threads_per_run,
                  total_outputs_len_filter_yoffset,
                  (double)print_interval *
                      KernelFilterYOffset::threads_per_run /
                      total_outputs_len_filter_yoffset,
                  print_interval * KernelFilterYOffset::threads_per_run /
                      time_yoffset * 1e-9);
      std::printf("seed_1     - %9.3f ms | %6.3f %% | %12" PRIu64
                  "                "
                  " |                 "
                  " | %7.3f Msps\n",
                  time_seed_1 * 1e3, time_seed_1 / time_total * 100.0,
                  total_outputs_len_filter_yoffset,
                  total_outputs_len_filter_yoffset / time_seed_1 * 1e-6);
      std::printf(
          "f_coarse   - %9.3f ms | %6.3f %% | %12" PRIu64 " -> %12" PRIu64
          " | 1 in %11.3f"
          " | %7.3f Gsps\n",
          time_filter_0A_coarse * 1e3,
          time_filter_0A_coarse / time_total * 100.0,
          total_outputs_len_filter_yoffset, total_outputs_len_filter_0A_coarse,
          (double)total_outputs_len_filter_yoffset /
              total_outputs_len_filter_0A_coarse,
          total_outputs_len_filter_yoffset / time_filter_0A_coarse * 1e-9);
      std::printf(
          "filter_med - %9.3f ms | %6.3f %% | %12" PRIu64 " -> %12" PRIu64
          " | 1 in %11.3f"
          " | %7.3f Gsps\n",
          time_filter_0A_med * 1e3, time_filter_0A_med / time_total * 100.0,
          total_outputs_len_filter_0A_coarse, total_outputs_len_filter_0A_med,
          (double)total_outputs_len_filter_0A_coarse /
              total_outputs_len_filter_0A_med,
          total_outputs_len_filter_0A_coarse / time_filter_0A_med * 1e-9);
      std::printf("filter_3AC - %9.3f ms | %6.3f %% | %12" PRIu64
                  " -> %12" PRIu64 " | 1 in %11.3f"
                  " | %7.3f Gsps\n",
                  time_filter_3AC * 1e3, time_filter_3AC / time_total * 100.0,
                  total_outputs_len_filter_0A_med, total_outputs_len_filter_3AC,
                  (double)total_outputs_len_filter_0A_med /
                      total_outputs_len_filter_3AC,
                  total_outputs_len_filter_0A_med / time_filter_3AC * 1e-9);
      std::printf("filter_3AR - %9.3f ms | %6.3f %% | %12" PRIu64
                  " -> %12" PRIu64 " | 1 in %11.3f"
                  " | %7.3f Gsps\n",
                  time_filter_3AR * 1e3, time_filter_3AR / time_total * 100.0,
                  total_outputs_len_filter_3AC, total_outputs_len_filter_3AR,
                  (double)total_outputs_len_filter_3AC /
                      total_outputs_len_filter_3AR,
                  total_outputs_len_filter_3AC / time_filter_3AR * 1e-9);
      std::printf("filter_3BC - %9.3f ms | %6.3f %% | %12" PRIu64
                  " -> %12" PRIu64 " | 1 in %11.3f"
                  " | %7.3f Gsps\n",
                  time_filter_3BC * 1e3, time_filter_3BC / time_total * 100.0,
                  total_outputs_len_filter_3AR, total_outputs_len_filter_3BC,
                  (double)total_outputs_len_filter_3AR /
                      total_outputs_len_filter_3BC,
                  total_outputs_len_filter_3AR / time_filter_3BC * 1e-9);
      std::printf("filter_3BR - %9.3f ms | %6.3f %% | %12" PRIu64
                  " -> %12" PRIu64 " | 1 in %11.3f"
                  " | %7.3f Gsps\n",
                  time_filter_3BR * 1e3, time_filter_3BR / time_total * 100.0,
                  total_outputs_len_filter_3BC, total_outputs_len_filter_3BR,
                  (double)total_outputs_len_filter_3BC /
                      total_outputs_len_filter_3BR,
                  total_outputs_len_filter_3BC / time_filter_3BR * 1e-9);
      std::printf("filter_isl - %9.3f ms | %6.3f %% | %12" PRIu64
                  " -> %12" PRIu64 " | 1 in %11.3f"
                  " | %7.3f Gsps\n",
                  time_filter_island * 1e3,
                  time_filter_island / time_total * 100.0,
                  total_outputs_len_filter_3BR, total_outputs_len_filter_island,
                  (double)total_outputs_len_filter_3BR /
                      total_outputs_len_filter_island,
                  total_outputs_len_filter_3BR / time_filter_island * 1e-9);

      std::printf(
          "total      - %9.3f ms | %6.3f %% |                             "
          " |                 "
          " | %7.3f Gsps ",
          time_total * 1e3, kernel_time_total / time_total * 100.0,
          print_interval * KernelFilterYOffset::threads_per_run / time_total *
              1e-9);
      size_t gpu_outputs_size;
      {
        std::lock_guard lock(outputs.mutex);
        gpu_outputs_size = outputs.queue.size();
      }
      std::printf("gpu_outputs.size() = %zu\n", gpu_outputs_size);

      start = end;
      time_yoffset = 0.0;
      time_seed_1 = 0.0;
      time_filter_0A_coarse = 0.0;
      time_filter_0A_med = 0.0;
      time_filter_3AR = 0.0;
      time_filter_3BC = 0.0;
      time_filter_3AC = 0.0;
      time_filter_3BR = 0.0;
      time_filter_island = 0.0;

      inputs_seed_1 = 0;
      inputs_yoffset = 0;
      inputs_filter_0A_coarse = 0;
      total_outputs_len_filter_yoffset = 0;
      total_outputs_len_filter_0A_coarse = 0;
      total_outputs_len_filter_0A_med = 0;
      total_outputs_len_filter_3BC = 0;
      total_outputs_len_filter_3AC = 0;
      total_outputs_len_filter_3AR = 0;
      total_outputs_len_filter_3BR = 0;
      total_outputs_len_filter_island = 0;
    }
  }
  TRY_CUDA(cudaFree(device_buffer_lens));
}
