#include "cpu.h"
#include "cubiomes.h"

#include <chrono>
#include <optional>

std::optional<CpuOutput> process(Cubiomes *cubiomes, const GpuOutput &input) {
  // return {{ input.seed, input.x, input.z, 0 }};
  // cubiomes_apply_seed(cubiomes, input.seed);

  //printf("unfiltered seed: %lld x: %d z: %d\n", input.seed, input.x, input.z);
  cubiomes_apply_climate(cubiomes, input.seed, 10);
  if (!cubiomes_is_surrounded(cubiomes, input.x, input.z, 40, -1.025)) {
    return {};
  }

  cubiomes_apply_climate(cubiomes, input.seed, 18);
  if (!cubiomes_is_surrounded(cubiomes, input.x, input.z, 4, -1.05)) {
    return {};//{{input.seed, input.x, input.z, 40}};
  }

  int max_cont = cubiomes_locate_climate_extreme(cubiomes, input.x/4, input.z/4, 128); 
  printf("1:4 seed: %lld x: %d z: %d\n", input.seed, input.x, input.z);
  return {{input.seed, input.x, input.z, max_cont}};
}

CpuThread::CpuThread(int id, GpuOutputs &inputs, CpuOutputs &outputs)
    : Thread(), id(id), inputs(inputs), outputs(outputs) {
  start();
}

void CpuThread::run() {
  std::printf("Started cpu thread %d\n", id);

  Cubiomes *cubiomes = cubiomes_create(large_biomes);

  while (!should_stop()) {
    GpuOutput input;
    {
      std::unique_lock lock(inputs.mutex);
      if (inputs.queue.empty()) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      input = inputs.queue.front();
      inputs.queue.pop();
    }

    const auto start = std::chrono::steady_clock::now();

    const auto output = process(cubiomes, input);

    const auto end = std::chrono::steady_clock::now();
    double time_total =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() *
        1e-9;
    // std::printf("Cpu test took %.3f s\n", time_total);

    if (!output)
      continue;

    {
      std::lock_guard lock(outputs.mutex);
      outputs.queue.push(output.value());
    }
  }

  cubiomes_free(cubiomes);
}
