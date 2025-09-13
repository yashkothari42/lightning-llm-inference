#include <string>
#include <print>
#include <filesystem>
#include <fstream>
#include <array>
#include <vector>
#include <tuple>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <map>
#include <variant>

// Helper functions for reading from fstream
template<typename T>
T read_value(std::fstream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

template<typename T, size_t N>
std::array<T, N> read_array(std::fstream& file) {
    std::array<T, N> arr;
    file.read(reinterpret_cast<char*>(arr.data()), sizeof(T) * N);
    return arr;
}

std::string read_string(std::fstream& file) {
    uint64_t length = read_value<uint64_t>(file);
    std::string str(length, '\0');
    file.read(str.data(), length);
    return str;
}

// Placeholder types and functions (you mentioned you'll remove manually)
enum gguf_type { GGUF_ARRAY = 0 };
enum gguf_metadata_value_type { GGUF_METADATA_ARRAY = 0 };
enum ggml_type { F32 = 0 };

struct GGUFMetadata {
    struct ArrayValue {};
    struct ScalarValue {};
    std::map<std::string, std::variant<ArrayValue, ScalarValue>> key_values;
    
    static ArrayValue create_array_value(gguf_type type) { return ArrayValue{}; }
};

void push_to_array(GGUFMetadata::ArrayValue& arr, GGUFMetadata::ScalarValue val) {}
GGUFMetadata::ScalarValue read_gguf_scalar_value_mmap(std::fstream& file, gguf_type type) { return GGUFMetadata::ScalarValue{}; }
ggml_type convert_raw_to_ggml_type(uint32_t raw) { return F32; }
uint32_t GGML_PAD(uint32_t offset, uint32_t align) { return ((offset + align - 1) / align) * align; }

#define LOG_SUCCESS(msg) std::println("SUCCESS: {}", msg)
#define LOG_PROGRESS(msg) std::println("PROGRESS: {}", msg)

int main(){
    std::string file_path = "/Users/kothari/Library/Caches/llama.cpp/Qwen_Qwen3-4B-GGUF_Qwen3-4B-Q4_K_M.gguf";

    auto file = std::fstream(file_path, std::ios::binary | std::ios::in);
    auto file_size = std::filesystem::file_size(file_path);
    std::println("File opened with size {}B", file_size);

    std::array<char, 4> magic = read_array<char, 4>(file);
    assert(magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F');
    
    
    // read version
    uint32_t version = read_value<uint32_t>(file);
    assert(version == 3);
    
    // read n_tensors
    int64_t n_tensors = read_value<int64_t>(file);
    assert(n_tensors == 398);
    
    // read n_kv
    int64_t n_kv = read_value<int64_t>(file);
    assert(n_kv == 28);
    
    // Create metadata structure
    GGUFMetadata metadata;
    
    auto metadata_start = std::chrono::high_resolution_clock::now();
    // read key-value pairs
    for (int64_t i = 0; i < n_kv; i++) {
        std::string key_string = read_string(file);
        gguf_type type = read_value<gguf_type>(file);
        
        if (type == gguf_type::GGUF_ARRAY) {
            gguf_type type_of_array = read_value<gguf_type>(file);
            int64_t size_of_array = read_value<int64_t>(file);
            
            GGUFMetadata::ArrayValue array = GGUFMetadata::create_array_value(type_of_array);
            
            for (int64_t j = 0; j < size_of_array; j++) {
                GGUFMetadata::ScalarValue value = read_gguf_scalar_value_mmap(file, type_of_array);
                push_to_array(array, value);
            }
            
            metadata.key_values[key_string] = array;
            
        } else {
            metadata.key_values[key_string] = read_gguf_scalar_value_mmap(file, type);
        }
    }
    auto metadata_end = std::chrono::high_resolution_clock::now();
    auto metadata_duration = std::chrono::duration_cast<std::chrono::milliseconds>(metadata_end - metadata_start);
    LOG_SUCCESS(std::format("Metadata reading completed in {} ms", metadata_duration.count()));
    
    // read tensor metadata
    LOG_PROGRESS("Reading tensor metadata...");
    std::vector<std::tuple<std::string, std::vector<int64_t>, ggml_type, uint64_t>> tensors;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const auto& name = read_string(file);
        uint32_t n_dims = read_value<uint32_t>(file);
        
        std::vector<int64_t> dims;
        dims.resize(n_dims);
        for(uint16_t dim = 0; dim < n_dims; dim++){
            dims[dim] = read_value<int64_t>(file);
        }
        uint32_t type_raw = read_value<uint32_t>(file);
        ggml_type type = convert_raw_to_ggml_type(type_raw);
        uint64_t offset = read_value<uint64_t>(file);
        tensors.push_back(std::make_tuple(name, dims, type, offset));
    }

    // pad so that the offset is a multiple of 32
    uint32_t data_start = GGML_PAD(static_cast<uint32_t>(file.tellg()), 32);
    assert(data_start == 0x5AE300);

 
    file.close();
    
    return 0;
}