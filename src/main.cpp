#include "common.h"
#include "cpu.h"
#include "gpu.h"

#include <charconv>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>

bool check_duplicate(bool duplicate, const char *option) {
  if (duplicate) {
    std::fprintf(stderr, "duplicate %s option\n", option);
    return true;
  }
  return false;
}

bool check_argument(int argc, int i, const char *option) {
  if (i >= argc) {
    std::fprintf(stderr, "missing argument to %s\n", option);
    return true;
  }
  return false;
}

struct Args {
  std::vector<int> devices;
  std::optional<int> threads;
  std::optional<std::string> output_file;

  bool parse(int argc, const char **const argv) {
    for (int i = 1; i < argc;) {
      const char *arg = argv[i++];

      if (std::strcmp("--device", arg) == 0) {
        if (check_argument(argc, i, arg))
          return false;
        const char *devices_str = argv[i++];
        const char *last = devices_str + std::strlen(devices_str);
        const char *first = devices_str;
        while (first != last) {
          int device;
          auto [ptr, ec] = std::from_chars(first, last, device, 10);
          if (ec != std::errc() || device < 0 ||
              std::find(devices.begin(), devices.end(), device) !=
                  devices.end() ||
              ptr != last && *ptr != ',') {
            std::fprintf(stderr, "invalid argument to --device: %s\n",
                         devices_str);
            return false;
          }
          devices.push_back(device);
          first = ptr;
          if (first != last)
            first++;
        }
      } else if (std::strcmp("--threads", arg) == 0) {
        if (check_duplicate((bool)threads, arg))
          return false;
        if (check_argument(argc, i, arg))
          return false;
        int threads_val = std::atoi(argv[i++]);
        if (threads_val <= 0 || threads_val > 1024) {
          std::fprintf(stderr, "invalid argument to --threads: %d\n",
                       threads_val);
          return false;
        }
        threads = threads_val;
      } else if (std::strcmp("--output", arg) == 0) {
        if (check_duplicate((bool)output_file, arg))
          return false;
        if (check_argument(argc, i, arg))
          return false;
        output_file = argv[i++];
      } else {
        std::fprintf(stderr, "unknown option: %s\n", arg);
        return false;
      }
    }

    if (devices.empty()) {
      devices.push_back(0);
    }

    return true;
  }
};

int main_inner(int argc, char **argv) {
  Args args{};
  if (!args.parse(argc, const_cast<const char **const>(argv))) {
    std::printf("Usage:\n%s [--device <device>,<device>,...] [--threads "
                "<threads>] [--output <output_file>]\n",
                argv[0]);
    return 1;
  }

  const int threads = args.threads.value_or(1);

  std::printf("Hello! large_biomes = %s\n", large_biomes ? "true" : "false");

  std::FILE *output_file = nullptr;
  if (threads != 0) {
    const char *output_file_path =
        args.output_file ? args.output_file.value().c_str() : "output.txt";
    output_file = std::fopen(output_file_path, "a");
    if (output_file == nullptr) {
      std::fprintf(stderr, "Could not open %s\n", output_file_path);
      return 1;
    }
    std::fprintf(output_file, "\n");
    std::fflush(output_file);
  }

  GpuOutputs gpu_outputs;
  CpuOutputs cpu_outputs;

  std::vector<std::unique_ptr<GpuThread>> gpu_threads;
  for (int device : args.devices) {
    gpu_threads.emplace_back(
        std::make_unique<GpuThread>(device, std::ref(gpu_outputs)));
  }

  std::vector<std::unique_ptr<CpuThread>> cpu_threads;
  for (int i = 0; i < threads; i++) {
    cpu_threads.emplace_back(std::make_unique<CpuThread>(
        i, std::ref(gpu_outputs), std::ref(cpu_outputs)));
  }

  for (size_t i = 0;; i++) {
    if (threads != 0) {
      std::lock_guard lock(cpu_outputs.mutex);
      while (!cpu_outputs.queue.empty()) {
        auto output = cpu_outputs.queue.front();
        cpu_outputs.queue.pop();
        std::printf("%" PRIi64 " at %" PRIi32 " %" PRIi32 " with %" PRIi32 "\n",
                 output.seed, output.x, output.z, output.score);
        std::fprintf(output_file,
                     "%" PRIi64 " %" PRIi32 " %" PRIi32 " %" PRIi32 "\n",
                     output.seed, output.x, output.z, output.score);
        std::fflush(output_file);
      }
    }

    if (args.devices.size() == 0 && i % 10 == 0) {
      std::lock_guard lock(gpu_outputs.mutex);
      std::printf("gpu_outputs.queue.size() = %zu\n", gpu_outputs.queue.size());
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  for (auto &thread : gpu_threads) {
    (*thread).stop();
  }
  for (auto &thread : cpu_threads) {
    (*thread).stop();
  }

  for (auto &thread : gpu_threads) {
    (*thread).join();
  }
  for (auto &thread : cpu_threads) {
    (*thread).join();
  }

  if (output_file != nullptr) {
    std::fclose(output_file);
  }
}

int main(int argc, char **argv) {
  try {
    main_inner(argc, argv);
  } catch (std::exception &e) {
    std::fprintf(stderr, "Uncaught exception in main: %s\n", e.what());
    std::abort();
  }
}
