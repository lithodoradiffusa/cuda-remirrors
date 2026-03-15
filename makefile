CUBIOMES_SRC := $(addprefix cubiomes/,biomenoise.c biomes.c finders.c generator.c layers.c noise.c)

# override with `make LARGE_BIOMES=1`
LARGE_BIOMES ?= 0
override CFLAGS += -O3
override CXXFLAGS += -O3 -std=c++20 -DMIRRORS_LARGE_BIOMES=$(LARGE_BIOMES)
override NVCC_FLAGS += $(CXXFLAGS) --expt-relaxed-constexpr --default-stream per-thread
# override NVCC_FLAGS += -arch=sm_61
#Windows compiles everything in one command
ifeq ($(OS),Windows_NT)

# CUDA_INC := "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
# CUDA_CRT := "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\crt"

CCBIN := "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

all: main.exe

# nvcc src/*.cpp src/*.c src/*.cu -o main.exe cubiomes/biomenoise.c cubiomes/biomes.c cubiomes/finders.c cubiomes/generator.c cubiomes/layers.c cubiomes/noise.c -arch=native -O3 -std=c++20 -DMIRRORS_LARGE_BIOMES=0 --expt-relaxed-constexpr --default-stream per-thread -D_WIN32_WINNT=0x0601
main.exe: src/*.*
	nvcc src/*.cpp src/*.c src/*.cu $(CUBIOMES_SRC) -o $@ $(NVCC_FLAGS) -D_WIN32_WINNT=0x0601 -ccbin $(CCBIN)
# -ccbin $(CCBIN) -I$(CUDA_INC) -I$(CUDA_CRT)
else # for linux/mac
override NVCC_FLAGS += -ccbin $(CXX)

MAIN_SRC := src/main.cpp
MAIN_DEP := $(MAIN_SRC) src/common.h

MAIN_SRC += gpu.o
MAIN_DEP += gpu.o src/gpu.h
MAIN_CXX := nvcc
MAIN_CXXFLAGS += $(NVCC_FLAGS)

MAIN_SRC += cpu.o cubiomes.o ringchecker_step.o libcubiomes.a
MAIN_DEP += cpu.o cubiomes.o ringchecker_step.o libcubiomes.a src/cpu.h


all: main

libcubiomes.a:
	$(CC) -c $(CUBIOMES_SRC) -fwrapv $(CFLAGS)
	$(AR) rcs libcubiomes.a biomenoise.o biomes.o finders.o generator.o layers.o noise.o

ringchecker_step.o: src/ringchecker_step.c src/ringchecker_step.h
	$(CC) -c $< -o $@ $(CFLAGS)

cubiomes.o: src/cubiomes.c src/cubiomes.h
	$(CC) -c $< -o $@ $(CFLAGS)

gpu.o: src/gpu.cu src/gpu.h src/common.h src/Random.h
	nvcc -c $< -o $@ $(NVCC_FLAGS)

cpu.o: src/cpu.cpp src/cpu.h src/common.h src/cubiomes.h
	$(CXX) -c $< -o $@ $(CXXFLAGS)

main: $(MAIN_DEP)
	$(MAIN_CXX) $(MAIN_SRC) -o $@ $(MAIN_CXXFLAGS)
endif
