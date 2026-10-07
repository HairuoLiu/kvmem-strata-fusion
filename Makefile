CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Werror -pedantic -I csrc -march=armv8.5-a+simd
LDFLAGS ?=

BUILD_DIR = build
SRCS = csrc/ubba_solver.cpp csrc/ifr_lse.cpp csrc/main_bench.cpp
OBJS = $(BUILD_DIR)/ubba_solver.o $(BUILD_DIR)/ifr_lse.o $(BUILD_DIR)/main_bench.o
TARGET = $(BUILD_DIR)/kvmem_engine_bench

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: csrc/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

bench: $(TARGET)
	./$(TARGET)

.PHONY: all clean bench
