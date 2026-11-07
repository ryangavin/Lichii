// SPDX-License-Identifier: GPL-3.0-or-later
// Generic Vulkan compute A/B runner: dispatches a baseline and a candidate SPIR-V module on
// identical inputs, compares the storage buffers they write, and times both with GPU
// timestamps. Bindings come from SPIR-V reflection. See tools/nx2/shader-ab.md.
// Build: c++ -std=c++20 -O2 tools/shader-ab/shader_ab.cpp -lvulkan -o build/shader-ab/shader-ab

#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string{operation} + ": VkResult=" + std::to_string(result));
}

std::vector<uint32_t> ReadSpirv(const std::string& path) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file || file.tellg() < 20 || file.tellg() % 4 != 0)
        throw std::runtime_error("Invalid SPIR-V file: " + path);
    std::vector<uint32_t> code(static_cast<size_t>(file.tellg()) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), code.size() * 4);
    if (!file || code[0] != 0x07230203)
        throw std::runtime_error("Invalid SPIR-V header: " + path);
    return code;
}

std::vector<uint8_t> ReadBytes(const std::string& path) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file) throw std::runtime_error("Cannot read " + path);
    std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    return bytes;
}

// ---- SPIR-V reflection -------------------------------------------------------------------

enum class Kind { Uniform, Storage, Image, Sampler, CombinedImage, TexelBuffer, StorageTexel };

const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Uniform: return "uniform";
    case Kind::Storage: return "storage";
    case Kind::Image: return "image";
    case Kind::Sampler: return "sampler";
    case Kind::CombinedImage: return "combined-image-sampler";
    case Kind::TexelBuffer: return "uniform-texel-buffer";
    case Kind::StorageTexel: return "storage-texel-buffer";
    }
    return "?";
}

struct Binding {
    uint32_t set{}, binding{}, count{1};
    Kind kind{};
    std::string name;
};

struct Reflection {
    std::string entry;
    std::array<uint32_t, 3> workgroup{};
    std::vector<Binding> bindings;
    bool push_constants{};
};

Reflection Reflect(const std::vector<uint32_t>& code, const std::string& wanted_entry) {
    std::map<uint32_t, std::string> names;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> pointers; // id -> storage class, pointee
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> arrays;   // id -> element, length id
    std::map<uint32_t, uint32_t> constants;                      // id -> 32-bit value
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> images;    // id -> dim, sampled
    std::set<uint32_t> buffer_blocks, samplers, sampled_images;
    std::map<uint32_t, uint32_t> sets, bindings;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> variables; // id -> type, storage class
    std::vector<std::pair<uint32_t, std::string>> entries;      // id, name (GLCompute only)
    std::map<uint32_t, std::array<uint32_t, 3>> local_sizes, local_size_ids;
    for (size_t offset = 5; offset < code.size();) {
        const uint32_t size = code[offset] >> 16, opcode = code[offset] & 0xffffU;
        if (size == 0 || offset + size > code.size()) throw std::runtime_error("Malformed SPIR-V");
        const uint32_t* w = &code[offset];
        if (opcode == 54) break; // OpFunction: all declarations precede it.
        switch (opcode) {
        case 5: // OpName
            names[w[1]] = reinterpret_cast<const char*>(&w[2]);
            break;
        case 15: // OpEntryPoint
            if (w[1] == 5) entries.emplace_back(w[2], reinterpret_cast<const char*>(&w[3]));
            break;
        case 16: // OpExecutionMode LocalSize
            if (w[2] == 17 && size >= 6) local_sizes[w[1]] = {w[3], w[4], w[5]};
            break;
        case 331: // OpExecutionModeId LocalSizeId
            if (w[2] == 38 && size >= 6) local_size_ids[w[1]] = {w[3], w[4], w[5]};
            break;
        case 25: images[w[1]] = {w[3], w[7]}; break;      // OpTypeImage: dim, sampled
        case 26: samplers.insert(w[1]); break;             // OpTypeSampler
        case 27: sampled_images.insert(w[1]); break;       // OpTypeSampledImage
        case 28: arrays[w[1]] = {w[2], w[3]}; break;       // OpTypeArray
        case 32: pointers[w[1]] = {w[2], w[3]}; break;     // OpTypePointer
        case 43: if (size >= 4) constants[w[2]] = w[3]; break; // OpConstant (first word)
        case 59: variables[w[2]] = {w[1], w[3]}; break;    // OpVariable
        case 71: // OpDecorate
            if (w[2] == 3) buffer_blocks.insert(w[1]);
            else if (w[2] == 33) bindings[w[1]] = w[3];
            else if (w[2] == 34) sets[w[1]] = w[3];
            break;
        default: break;
        }
        offset += size;
    }
    if (entries.empty()) throw std::runtime_error("Module has no GLCompute entry point");
    Reflection result;
    uint32_t entry_id = entries.front().first;
    result.entry = entries.front().second;
    if (!wanted_entry.empty()) {
        const auto it = std::find_if(entries.begin(), entries.end(),
                                     [&](const auto& e) { return e.second == wanted_entry; });
        if (it == entries.end()) throw std::runtime_error("No compute entry point named " + wanted_entry);
        entry_id = it->first;
        result.entry = it->second;
    }
    if (local_sizes.contains(entry_id)) {
        result.workgroup = local_sizes.at(entry_id);
    } else if (local_size_ids.contains(entry_id)) {
        for (size_t i = 0; i < 3; ++i) {
            const auto id = local_size_ids.at(entry_id)[i];
            result.workgroup[i] = constants.contains(id) ? constants.at(id) : 0;
        }
    }
    for (const auto& [id, var] : variables) {
        const auto [type, storage] = var;
        if (storage == 9) { // PushConstant
            result.push_constants = true;
            continue;
        }
        if (storage != 0 && storage != 2 && storage != 12) continue;
        if (!pointers.contains(type)) continue;
        uint32_t pointee = pointers.at(type).second;
        Binding b;
        b.name = names.contains(id) ? names.at(id) : "";
        while (arrays.contains(pointee)) {
            const auto [element, length] = arrays.at(pointee);
            b.count *= constants.contains(length) ? constants.at(length) : 1;
            pointee = element;
        }
        if (storage == 12) b.kind = Kind::Storage;
        else if (storage == 2) b.kind = buffer_blocks.contains(pointee) ? Kind::Storage : Kind::Uniform;
        else if (images.contains(pointee)) {
            const auto [dim, sampled] = images.at(pointee);
            if (dim == 5) b.kind = sampled == 2 ? Kind::StorageTexel : Kind::TexelBuffer;
            else b.kind = Kind::Image;
        } else if (samplers.contains(pointee)) b.kind = Kind::Sampler;
        else if (sampled_images.contains(pointee)) b.kind = Kind::CombinedImage;
        else continue;
        if (!sets.contains(id) || !bindings.contains(id))
            throw std::runtime_error("Descriptor variable without set/binding: " + b.name);
        b.set = sets.at(id);
        b.binding = bindings.at(id);
        result.bindings.push_back(b);
    }
    std::sort(result.bindings.begin(), result.bindings.end(), [](const auto& a, const auto& b) {
        return std::tie(a.set, a.binding) < std::tie(b.set, b.binding);
    });
    // The recompiler aliases typed views (u32, u32x2, u32x4) of one buffer on one binding.
    std::vector<Binding> merged;
    for (const auto& b : result.bindings) {
        if (!merged.empty() && merged.back().set == b.set && merged.back().binding == b.binding) {
            if (merged.back().kind != b.kind)
                throw std::runtime_error("Set " + std::to_string(b.set) + " binding " + std::to_string(b.binding) +
                                         " is declared with different descriptor kinds");
            merged.back().count = std::max(merged.back().count, b.count);
            continue;
        }
        merged.push_back(b);
    }
    result.bindings = std::move(merged);
    return result;
}

// ---- Case description --------------------------------------------------------------------

enum class Fill { Zero, Sentinel, Index, Random, RandomF32, RandomHalf, File };
enum class Compare { None, Bitwise, UlpF32, UlpF16, AbsF32 };

const char* FillName(Fill fill) {
    constexpr std::array names{"zero", "sentinel", "index", "random", "random-f32", "random-half", "file"};
    return names[static_cast<size_t>(fill)];
}

const char* CompareName(Compare compare) {
    constexpr std::array names{"none", "bitwise", "f32-ulp", "f16-ulp", "f32-abs"};
    return names[static_cast<size_t>(compare)];
}

struct BufferSpec {
    uint32_t set{}, binding{};
    uint64_t size{65536};
    Fill fill{Fill::Sentinel};
    std::string file;
    Compare compare{Compare::Bitwise};
    double tolerance{};
    uint64_t begin{}, end{UINT64_MAX};
    bool explicit_fill{}, explicit_compare{};
};

struct Options {
    std::string baseline, candidate, entry, json;
    std::array<uint32_t, 3> dispatch{1, 1, 1};
    uint32_t subgroup{}, device{}, cases{64}, timing{}, samples{10}, show{5};
    bool validate{}, reflect_only{};
    std::vector<BufferSpec> buffers;
    std::vector<uint8_t> push;
};

std::map<std::string, std::string> KeyValues(const std::string& text) {
    std::map<std::string, std::string> result;
    size_t start = 0;
    while (start <= text.size()) {
        const auto comma = text.find(',', start);
        const auto item = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        const auto eq = item.find('=');
        if (!item.empty()) {
            if (eq == std::string::npos) result[item] = "";
            else result[item.substr(0, eq)] = item.substr(eq + 1);
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return result;
}

BufferSpec ParseBuffer(const std::string& text) {
    BufferSpec spec;
    for (const auto& [key, value] : KeyValues(text)) {
        if (key == "set") spec.set = std::stoul(value);
        else if (key == "binding") spec.binding = std::stoul(value);
        else if (key == "size") spec.size = std::stoull(value);
        else if (key == "fill") {
            spec.explicit_fill = true;
            if (value == "zero") spec.fill = Fill::Zero;
            else if (value == "sentinel") spec.fill = Fill::Sentinel;
            else if (value == "index") spec.fill = Fill::Index;
            else if (value == "random") spec.fill = Fill::Random;
            else if (value == "random-f32") spec.fill = Fill::RandomF32;
            else if (value == "random-half") spec.fill = Fill::RandomHalf;
            else throw std::runtime_error("Unknown fill: " + value);
        } else if (key == "file") {
            spec.explicit_fill = true;
            spec.fill = Fill::File;
            spec.file = value;
        } else if (key == "compare") {
            spec.explicit_compare = true;
            if (value == "none") spec.compare = Compare::None;
            else if (value == "bitwise") spec.compare = Compare::Bitwise;
            else if (value == "f32-ulp") spec.compare = Compare::UlpF32;
            else if (value == "f16-ulp") spec.compare = Compare::UlpF16;
            else if (value == "f32-abs") spec.compare = Compare::AbsF32;
            else throw std::runtime_error("Unknown compare mode: " + value);
        } else if (key == "tolerance") spec.tolerance = std::stod(value);
        else if (key == "begin") spec.begin = std::stoull(value);
        else if (key == "end") spec.end = std::stoull(value);
        else throw std::runtime_error("Unknown buffer option: " + key);
    }
    if (spec.size == 0 || spec.size % 4) throw std::runtime_error("Buffer size must be a positive multiple of 4");
    return spec;
}

std::vector<uint8_t> ParseHex(const std::string& hex) {
    if (hex.size() % 2) throw std::runtime_error("Odd-length hex string");
    std::vector<uint8_t> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<uint8_t>(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
    return bytes;
}

Options ParseOptions(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--baseline") o.baseline = value();
        else if (flag == "--candidate") o.candidate = value();
        else if (flag == "--entry") o.entry = value();
        else if (flag == "--json") o.json = value();
        else if (flag == "--device") o.device = std::stoul(value());
        else if (flag == "--subgroup") o.subgroup = std::stoul(value());
        else if (flag == "--cases") o.cases = std::stoul(value());
        else if (flag == "--timing") o.timing = std::stoul(value());
        else if (flag == "--samples") o.samples = std::stoul(value());
        else if (flag == "--show") o.show = std::stoul(value());
        else if (flag == "--validate") o.validate = true;
        else if (flag == "--reflect") o.reflect_only = true;
        else if (flag == "--dispatch") {
            if (std::sscanf(value().c_str(), "%u,%u,%u", &o.dispatch[0], &o.dispatch[1], &o.dispatch[2]) != 3)
                throw std::runtime_error("--dispatch expects X,Y,Z workgroups");
        } else if (flag == "--buffer") o.buffers.push_back(ParseBuffer(value()));
        else if (flag == "--push") {
            const auto kv = KeyValues(value());
            if (kv.contains("hex")) o.push = ParseHex(kv.at("hex"));
            else if (kv.contains("file")) o.push = ReadBytes(kv.at("file"));
            else throw std::runtime_error("--push expects hex=... or file=...");
        } else throw std::runtime_error("Unknown option: " + flag);
    }
    if (o.baseline.empty()) throw std::runtime_error("--baseline is required");
    if (!o.reflect_only && o.candidate.empty()) throw std::runtime_error("--candidate is required");
    if (o.cases == 0 || o.samples < 2) throw std::runtime_error("Need at least one case and two samples");
    return o;
}

// ---- Deterministic inputs ----------------------------------------------------------------

uint32_t RandomHalfPair(std::mt19937& random, uint32_t schedule, uint32_t position) {
    constexpr std::array<uint16_t, 20> edges{0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
        0x3555, 0xb555, 0x3bff, 0x3c00, 0x3c01, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe01};
    const auto half = [&](uint32_t index) -> uint32_t {
        switch (schedule % 4) {
        case 0: return edges[(index + schedule / 4) % edges.size()];
        case 1: return random() & 0xffffU; // Any bit pattern, NaN and infinity included.
        case 2: return (random() & 0x83ffU) | ((8U + random() % 9U) << 10); // Normal, |x| < 2
        default: return random() & 0x87ffU; // Subnormal or the smallest normals
        }
    };
    const uint32_t low = half(position * 2);
    return low | (half(position * 2 + 1) << 16);
}

uint32_t RandomF32(std::mt19937& random, uint32_t schedule, uint32_t position) {
    constexpr std::array<uint32_t, 12> edges{0, 0x80000000U, 1, 0x00800000U, 0x3f800000U, 0xbf800000U,
        0x3f7fffffU, 0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U, 0x7fc00000U};
    switch (schedule % 4) {
    case 0: return edges[(position + schedule / 4) % edges.size()];
    case 1: return random();
    case 2: { // Finite values in roughly [-4096, 4096].
        float value;
        const uint32_t bits = (random() & 0x807fffffU) | ((115U + random() % 24U) << 23);
        std::memcpy(&value, &bits, 4);
        (void)value;
        return bits;
    }
    default: return (random() & 0x803fffffU) | (random() % 2 ? 0x00800000U : 0); // Tiny values
    }
}

std::vector<uint8_t> MakeInput(const BufferSpec& spec, uint32_t test, const std::vector<uint8_t>& file) {
    std::vector<uint8_t> data(spec.size);
    const uint32_t words = static_cast<uint32_t>(spec.size / 4);
    std::mt19937 random{0x5ab00000U + test * 977U + spec.set * 131U + spec.binding};
    auto* out = reinterpret_cast<uint32_t*>(data.data());
    switch (spec.fill) {
    case Fill::Zero: break;
    case Fill::Sentinel: std::fill_n(out, words, 0xcdcdcdcdU); break;
    case Fill::Index: for (uint32_t i = 0; i < words; ++i) out[i] = i; break;
    case Fill::Random: for (uint32_t i = 0; i < words; ++i) out[i] = random(); break;
    case Fill::RandomF32: for (uint32_t i = 0; i < words; ++i) out[i] = RandomF32(random, test, i); break;
    case Fill::RandomHalf: for (uint32_t i = 0; i < words; ++i) out[i] = RandomHalfPair(random, test, i); break;
    case Fill::File: std::memcpy(data.data(), file.data(), std::min(file.size(), data.size())); break;
    }
    return data;
}

// ---- Comparison --------------------------------------------------------------------------

int64_t Ordered(uint32_t bits, uint32_t sign_bit) {
    return bits & sign_bit ? -static_cast<int64_t>(bits & (sign_bit - 1)) : static_cast<int64_t>(bits);
}

bool IsNanF32(uint32_t b) { return (b & 0x7fffffffU) > 0x7f800000U; }
bool IsNanF16(uint32_t b) { return (b & 0x7fffU) > 0x7c00U; }

// Returns the distance metric for a mismatch, or nullopt when the elements agree.
std::optional<double> Differ(Compare mode, uint32_t a, uint32_t b, double tolerance) {
    if (a == b) return std::nullopt;
    switch (mode) {
    case Compare::Bitwise: return 1;
    case Compare::UlpF32: {
        if (IsNanF32(a) && IsNanF32(b)) return std::nullopt;
        if (IsNanF32(a) || IsNanF32(b)) return INFINITY;
        const auto d = std::abs(Ordered(a, 0x80000000U) - Ordered(b, 0x80000000U));
        return d <= tolerance ? std::nullopt : std::optional<double>(static_cast<double>(d));
    }
    case Compare::UlpF16: {
        if (IsNanF16(a) && IsNanF16(b)) return std::nullopt;
        if (IsNanF16(a) || IsNanF16(b)) return INFINITY;
        const auto d = std::abs(Ordered(a, 0x8000U) - Ordered(b, 0x8000U));
        return d <= tolerance ? std::nullopt : std::optional<double>(static_cast<double>(d));
    }
    case Compare::AbsF32: {
        float fa, fb;
        std::memcpy(&fa, &a, 4);
        std::memcpy(&fb, &b, 4);
        if (std::isnan(fa) && std::isnan(fb)) return std::nullopt;
        if (fa == fb) return std::nullopt;
        const double d = std::fabs(static_cast<double>(fa) - static_cast<double>(fb));
        return d <= tolerance ? std::nullopt : std::optional<double>(d);
    }
    case Compare::None: return std::nullopt;
    }
    return std::nullopt;
}

struct Mismatch {
    uint32_t test, set, binding;
    uint64_t offset;
    uint32_t baseline, candidate;
    double distance;
};

// ---- Vulkan runner -----------------------------------------------------------------------

struct DeviceBuffer {
    BufferSpec spec;
    Kind kind{};
    uint32_t count{1};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    uint64_t staging_offset{};
};

struct Runner {
    VkInstance instance{};
    VkDebugUtilsMessengerEXT messenger{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family{};
    VkCommandPool command_pool{};
    VkCommandBuffer command{};
    VkFence fence{};
    VkQueryPool queries{};
    float timestamp_period{};
    uint32_t timestamp_bits{};
    uint32_t subgroup_size{}, default_subgroup{};
    std::string device_name, driver_name;
    std::vector<VkDescriptorSetLayout> set_layouts;
    VkDescriptorPool descriptor_pool{};
    std::vector<VkDescriptorSet> descriptor_sets;
    VkPipelineLayout layout{};
    std::array<VkPipeline, 2> pipelines{};
    std::array<double, 2> pipeline_ms{};
    std::vector<DeviceBuffer> buffers;
    VkBuffer staging{};
    VkDeviceMemory staging_memory{};
    uint8_t* mapped{};
    uint64_t staging_size{};
    std::vector<uint8_t> push;
    uint32_t validation_errors{};
    VkPhysicalDeviceMemoryProperties memories{};

    ~Runner() {
        if (device) {
            vkDeviceWaitIdle(device);
            for (auto p : pipelines) if (p) vkDestroyPipeline(device, p, nullptr);
            if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
            if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            for (auto l : set_layouts) if (l) vkDestroyDescriptorSetLayout(device, l, nullptr);
            for (auto& b : buffers) {
                if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
                if (b.memory) vkFreeMemory(device, b.memory, nullptr);
            }
            if (mapped) vkUnmapMemory(device, staging_memory);
            if (staging) vkDestroyBuffer(device, staging, nullptr);
            if (staging_memory) vkFreeMemory(device, staging_memory, nullptr);
            if (queries) vkDestroyQueryPool(device, queries, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (messenger) {
            auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy) destroy(instance, messenger, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL Message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                  VkDebugUtilsMessageTypeFlagsEXT,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                  void* user) {
        if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            ++static_cast<Runner*>(user)->validation_errors;
            std::fprintf(stderr, "VALIDATION: %s\n", data->pMessage);
        }
        return VK_FALSE;
    }

    void CreateInstance(bool validate) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Lichii shader A/B";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create.pApplicationInfo = &app;
        const char* layer = "VK_LAYER_KHRONOS_validation";
        const char* extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        VkDebugUtilsMessengerCreateInfoEXT debug{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        if (validate) {
            uint32_t count{};
            vkEnumerateInstanceLayerProperties(&count, nullptr);
            std::vector<VkLayerProperties> layers(count);
            vkEnumerateInstanceLayerProperties(&count, layers.data());
            if (std::none_of(layers.begin(), layers.end(),
                             [&](const auto& l) { return std::strcmp(l.layerName, layer) == 0; }))
                throw std::runtime_error("VK_LAYER_KHRONOS_validation is not installed");
            create.enabledLayerCount = 1;
            create.ppEnabledLayerNames = &layer;
            create.enabledExtensionCount = 1;
            create.ppEnabledExtensionNames = &extension;
            debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
            debug.pfnUserCallback = Message;
            debug.pUserData = this;
            create.pNext = &debug;
        }
        Check(vkCreateInstance(&create, nullptr, &instance), "vkCreateInstance");
        if (validate) {
            auto make = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
            if (!make) throw std::runtime_error("Debug utils messenger unavailable");
            Check(make(instance, &debug, nullptr, &messenger), "create debug messenger");
        }
    }

    void CreateDevice(uint32_t device_index, uint32_t requested_subgroup) {
        uint32_t count{};
        Check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
        std::vector<VkPhysicalDevice> devices(count);
        Check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
        if (device_index >= count) throw std::runtime_error("Requested GPU index does not exist");
        physical = devices[device_index];
        VkPhysicalDeviceSubgroupSizeControlProperties size_props{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        driver.pNext = &size_props;
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        subgroup.pNext = &driver;
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physical, &props);
        if (props.properties.apiVersion < VK_API_VERSION_1_3)
            throw std::runtime_error("This runner requires a Vulkan 1.3 device");
        device_name = props.properties.deviceName;
        driver_name = std::string{driver.driverName} + " " + driver.driverInfo;
        timestamp_period = props.properties.limits.timestampPeriod;
        default_subgroup = subgroup.subgroupSize;
        // Enable every core feature the device reports so game-derived modules can use
        // 8/16/64-bit types, subgroup extended types and the like without a per-case list.
        uint32_t ext_count{};
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> avail_exts(ext_count);
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, avail_exts.data());
        const auto has_ext = [&](const char* name) {
            return std::any_of(avail_exts.begin(), avail_exts.end(),
                               [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
        };
        std::vector<const char*> enabled_exts;
        VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR wg_layout{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        if (has_ext(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME)) {
            enabled_exts.push_back(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
            f13.pNext = &wg_layout;
        }
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f12.pNext = &f13;
        f11.pNext = &f12;
        features.pNext = &f11;
        vkGetPhysicalDeviceFeatures2(physical, &features);
        f11.protectedMemory = VK_FALSE;
        features.features.robustBufferAccess = VK_TRUE; // Random inputs must not fault the GPU.
        if (requested_subgroup && (!f13.subgroupSizeControl ||
            !(size_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
            requested_subgroup < size_props.minSubgroupSize ||
            requested_subgroup > size_props.maxSubgroupSize))
            throw std::runtime_error("Requested compute subgroup size is unsupported on this device");
        subgroup_size = requested_subgroup;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
        family = 0;
        while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
        if (family == count) throw std::runtime_error("No compute queue");
        timestamp_bits = families[family].timestampValidBits;
        const float priority = 1;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.pNext = &features;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(enabled_exts.size());
        device_info.ppEnabledExtensionNames = enabled_exts.data();
        Check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, family, 0, &queue);
        vkGetPhysicalDeviceMemoryProperties(physical, &memories);
        VkCommandPoolCreateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_info.queueFamilyIndex = family;
        command_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        Check(vkCreateCommandPool(device, &command_info, nullptr, &command_pool), "create command pool");
        VkCommandBufferAllocateInfo command_allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_allocation.commandPool = command_pool;
        command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocation.commandBufferCount = 1;
        Check(vkAllocateCommandBuffers(device, &command_allocation, &command), "allocate command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        Check(vkCreateFence(device, &fence_info, nullptr, &fence), "create fence");
        if (timestamp_bits) {
            VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query_info.queryCount = 2;
            Check(vkCreateQueryPool(device, &query_info, nullptr, &queries), "create timestamp pool");
        }
    }

    uint32_t MemoryType(uint32_t allowed, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
        for (int pass = 0; pass < 2; ++pass) {
            const auto wanted = pass == 0 ? required | preferred : required;
            for (uint32_t i = 0; i < memories.memoryTypeCount; ++i)
                if ((allowed & (1U << i)) && (memories.memoryTypes[i].propertyFlags & wanted) == wanted)
                    return i;
        }
        throw std::runtime_error("No suitable memory type");
    }

    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                      VkMemoryPropertyFlags preferred, VkBuffer& buffer, VkDeviceMemory& memory) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(vkCreateBuffer(device, &info, nullptr, &buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = MemoryType(requirements.memoryTypeBits, required, preferred);
        Check(vkAllocateMemory(device, &allocation, nullptr, &memory), "vkAllocateMemory");
        Check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
    }

    // Builds descriptor layouts from the reflected interface plus the user's buffer specs.
    void CreateResources(const Reflection& reflection, const std::vector<BufferSpec>& specs,
                         std::vector<uint8_t> push_bytes) {
        push = std::move(push_bytes);
        uint32_t max_set = 0;
        for (const auto& b : reflection.bindings) {
            if (b.kind != Kind::Uniform && b.kind != Kind::Storage)
                throw std::runtime_error("Unsupported binding kind " + std::string{KindName(b.kind)} +
                                         " at set " + std::to_string(b.set) + " binding " +
                                         std::to_string(b.binding) + " (" + b.name + ")");
            max_set = std::max(max_set, b.set);
            DeviceBuffer buffer;
            buffer.kind = b.kind;
            buffer.count = b.count;
            buffer.spec.set = b.set;
            buffer.spec.binding = b.binding;
            if (b.kind == Kind::Uniform) {
                buffer.spec.fill = Fill::Zero;
                buffer.spec.compare = Compare::None;
            }
            for (const auto& spec : specs)
                if (spec.set == b.set && spec.binding == b.binding) {
                    const auto defaults = buffer.spec;
                    buffer.spec = spec;
                    if (!spec.explicit_fill) buffer.spec.fill = defaults.fill;
                    if (!spec.explicit_compare) buffer.spec.compare = defaults.compare;
                }
            buffers.push_back(buffer);
        }
        for (const auto& spec : specs) {
            if (std::none_of(buffers.begin(), buffers.end(), [&](const auto& b) {
                    return b.spec.set == spec.set && b.spec.binding == spec.binding; }))
                throw std::runtime_error("--buffer names set " + std::to_string(spec.set) + " binding " +
                                         std::to_string(spec.binding) + ", which neither module declares");
        }
        if (reflection.push_constants && push.empty())
            throw std::runtime_error("Module declares push constants; pass --push");
        staging_size = 0;
        for (auto& b : buffers) {
            b.staging_offset = staging_size;
            staging_size += (b.spec.size * b.count + 255) & ~uint64_t{255};
            CreateBuffer(b.spec.size * b.count,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b.buffer, b.memory);
        }
        if (staging_size == 0) staging_size = 256;
        CreateBuffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
                     staging, staging_memory);
        void* pointer{};
        Check(vkMapMemory(device, staging_memory, 0, staging_size, 0, &pointer), "vkMapMemory");
        mapped = static_cast<uint8_t*>(pointer);

        set_layouts.resize(max_set + 1);
        std::map<VkDescriptorType, uint32_t> pool_counts;
        for (uint32_t set = 0; set <= max_set; ++set) {
            std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
            for (const auto& b : buffers) {
                if (b.spec.set != set) continue;
                const auto type = b.kind == Kind::Uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                layout_bindings.push_back({b.spec.binding, type, b.count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
                pool_counts[type] += b.count;
            }
            VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
            info.pBindings = layout_bindings.data();
            Check(vkCreateDescriptorSetLayout(device, &info, nullptr, &set_layouts[set]), "create set layout");
        }
        std::vector<VkDescriptorPoolSize> sizes;
        for (const auto& [type, n] : pool_counts) sizes.push_back({type, n});
        if (sizes.empty()) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1});
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = max_set + 1;
        pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
        pool_info.pPoolSizes = sizes.data();
        Check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "create descriptor pool");
        descriptor_sets.resize(max_set + 1);
        VkDescriptorSetAllocateInfo set_allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_allocation.descriptorPool = descriptor_pool;
        set_allocation.descriptorSetCount = max_set + 1;
        set_allocation.pSetLayouts = set_layouts.data();
        Check(vkAllocateDescriptorSets(device, &set_allocation, descriptor_sets.data()), "allocate descriptor sets");
        std::vector<std::vector<VkDescriptorBufferInfo>> infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes;
        for (size_t i = 0; i < buffers.size(); ++i) {
            const auto& b = buffers[i];
            for (uint32_t element = 0; element < b.count; ++element)
                infos[i].push_back({b.buffer, element * b.spec.size, b.spec.size});
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = descriptor_sets[b.spec.set];
            write.dstBinding = b.spec.binding;
            write.descriptorCount = b.count;
            write.descriptorType = b.kind == Kind::Uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                           : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = infos[i].data();
            writes.push_back(write);
        }
        if (!writes.empty())
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push.size())};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = max_set + 1;
        layout_info.pSetLayouts = set_layouts.data();
        layout_info.pushConstantRangeCount = push.empty() ? 0 : 1;
        layout_info.pPushConstantRanges = &range;
        Check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout), "create pipeline layout");
    }

    void CreatePipeline(uint32_t index, const std::vector<uint32_t>& code, const std::string& entry) {
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = code.size() * 4;
        module_info.pCode = code.data();
        VkShaderModule module;
        Check(vkCreateShaderModule(device, &module_info, nullptr, &module), "create shader module");
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        size.requiredSubgroupSize = subgroup_size;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline.stage.pNext = subgroup_size ? &size : nullptr;
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = module;
        pipeline.stage.pName = entry.c_str();
        pipeline.layout = layout;
        const auto start = std::chrono::steady_clock::now();
        const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipelines[index]);
        pipeline_ms[index] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        vkDestroyShaderModule(device, module, nullptr);
        Check(result, "create compute pipeline");
    }

    // Uploads every buffer's host image, dispatches `repeats` times and downloads the results.
    // Returns the GPU time per dispatch in nanoseconds (0 without timestamp support).
    double Run(uint32_t index, const std::vector<std::vector<uint8_t>>& inputs,
               std::vector<std::vector<uint8_t>>& outputs, const std::array<uint32_t, 3>& dispatch,
               uint32_t repeats) {
        for (size_t i = 0; i < buffers.size(); ++i)
            std::memcpy(mapped + buffers[i].staging_offset, inputs[i].data(), inputs[i].size());
        Check(vkResetCommandBuffer(command, 0), "reset command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(vkBeginCommandBuffer(command, &begin), "begin command buffer");
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        const auto wait = [&](VkPipelineStageFlags src, VkAccessFlags src_access, VkPipelineStageFlags dst,
                              VkAccessFlags dst_access) {
            barrier.srcAccessMask = src_access;
            barrier.dstAccessMask = dst_access;
            vkCmdPipelineBarrier(command, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        };
        wait(VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_ACCESS_TRANSFER_READ_BIT);
        for (const auto& b : buffers) {
            const VkBufferCopy copy{b.staging_offset, 0, b.spec.size * b.count};
            vkCmdCopyBuffer(command, staging, b.buffer, 1, &copy);
        }
        const auto shader_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        wait(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
             shader_access);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[index]);
        if (!descriptor_sets.empty())
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                                    static_cast<uint32_t>(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
        if (!push.empty())
            vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(push.size()), push.data());
        if (queries) {
            vkCmdResetQueryPool(command, queries, 0, 2);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
        }
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            vkCmdDispatch(command, dispatch[0], dispatch[1], dispatch[2]);
            if (repeat + 1 < repeats)
                wait(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, shader_access);
        }
        if (queries) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        wait(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_ACCESS_TRANSFER_READ_BIT);
        for (const auto& b : buffers) {
            const VkBufferCopy copy{0, b.staging_offset, b.spec.size * b.count};
            vkCmdCopyBuffer(command, b.buffer, staging, 1, &copy);
        }
        wait(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
             VK_ACCESS_HOST_READ_BIT);
        Check(vkEndCommandBuffer(command), "end command buffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        Check(vkQueueSubmit(queue, 1, &submit, fence), "submit compute");
        const auto waited = vkWaitForFences(device, 1, &fence, VK_TRUE, 30'000'000'000ULL);
        if (waited == VK_TIMEOUT) throw std::runtime_error("GPU did not finish within 30 s: the shader may hang");
        Check(waited, "wait for compute");
        Check(vkResetFences(device, 1, &fence), "reset fence");
        outputs.resize(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            outputs[i].resize(inputs[i].size());
            std::memcpy(outputs[i].data(), mapped + buffers[i].staging_offset, outputs[i].size());
        }
        if (!queries) return 0;
        std::array<uint64_t, 2> times;
        Check(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(times), times.data(), sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "read timestamps");
        const uint64_t mask = timestamp_bits == 64 ? UINT64_MAX : (1ULL << timestamp_bits) - 1;
        return static_cast<double>((times[1] - times[0]) & mask) * timestamp_period / std::max(repeats, 1U);
    }
};

// ---- Reporting ---------------------------------------------------------------------------

std::string JsonString(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) { out += ' '; continue; }
        out += c;
    }
    return out + "\"";
}

std::string JsonNumber(double value) {
    if (std::isinf(value)) return "\"inf\"";
    if (std::isnan(value)) return "\"nan\"";
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

struct Stats {
    double min{}, median{}, p90{};
};

Stats Summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto at = [&](double fraction) {
        return values[std::min(values.size() - 1, static_cast<size_t>(fraction * values.size()))];
    };
    return {values.front(), (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2, at(0.9)};
}

std::string BindingsJson(const Reflection& r) {
    std::string out = "[";
    for (size_t i = 0; i < r.bindings.size(); ++i) {
        const auto& b = r.bindings[i];
        out += (i ? "," : "") + std::string{"{\"set\":"} + std::to_string(b.set) + ",\"binding\":" +
               std::to_string(b.binding) + ",\"count\":" + std::to_string(b.count) + ",\"kind\":" +
               JsonString(KindName(b.kind)) + ",\"name\":" + JsonString(b.name) + "}";
    }
    return out + "]";
}

std::string ReflectionJson(const Reflection& r) {
    return "{\"entry\":" + JsonString(r.entry) + ",\"workgroup\":[" + std::to_string(r.workgroup[0]) + "," +
           std::to_string(r.workgroup[1]) + "," + std::to_string(r.workgroup[2]) + "],\"push_constants\":" +
           (r.push_constants ? "true" : "false") + ",\"bindings\":" + BindingsJson(r) + "}";
}
} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s --baseline A.spv --candidate B.spv [--dispatch X,Y,Z] [--subgroup N]\n"
            "          [--cases N] [--timing REPEATS] [--samples N] [--device N] [--entry NAME]\n"
            "          [--buffer set=S,binding=B,size=BYTES,fill=MODE|file=PATH,compare=MODE,\n"
            "                   tolerance=T,begin=BYTE,end=BYTE]... [--push hex=..|file=PATH]\n"
            "          [--validate] [--json OUT] [--show N]\n"
            "       %s --reflect --baseline MODULE.spv\n"
            "Fills: zero sentinel index random random-f32 random-half. Compares: none bitwise\n"
            "f32-ulp f16-ulp f32-abs. Storage buffers default to sentinel fill and bitwise compare;\n"
            "uniform buffers default to zero fill and no compare.\n", argv[0], argv[0]);
        return 2;
    }
    const Options options = ParseOptions(argc, argv);
    const auto baseline_code = ReadSpirv(options.baseline);
    const auto baseline = Reflect(baseline_code, options.entry);
    if (options.reflect_only) {
        std::printf("%s\n", ReflectionJson(baseline).c_str());
        return 0;
    }
    const auto candidate_code = ReadSpirv(options.candidate);
    const auto candidate = Reflect(candidate_code, options.entry);
    if (baseline.bindings.size() != candidate.bindings.size())
        throw std::runtime_error("Modules declare different descriptor interfaces: " + BindingsJson(baseline) +
                                 " vs " + BindingsJson(candidate));
    for (size_t i = 0; i < baseline.bindings.size(); ++i) {
        const auto& a = baseline.bindings[i];
        const auto& b = candidate.bindings[i];
        if (a.set != b.set || a.binding != b.binding || a.kind != b.kind || a.count != b.count)
            throw std::runtime_error("Modules declare different descriptor interfaces: " + BindingsJson(baseline) +
                                     " vs " + BindingsJson(candidate));
    }
    if (baseline.workgroup != candidate.workgroup)
        std::fprintf(stderr, "WARNING: workgroup sizes differ (%u,%u,%u vs %u,%u,%u); the same dispatch "
                             "counts cover different invocations.\n", baseline.workgroup[0], baseline.workgroup[1],
                     baseline.workgroup[2], candidate.workgroup[0], candidate.workgroup[1], candidate.workgroup[2]);

    Runner runner;
    runner.CreateInstance(options.validate);
    runner.CreateDevice(options.device, options.subgroup);
    runner.CreateResources(baseline, options.buffers, options.push);
    runner.CreatePipeline(0, baseline_code, baseline.entry);
    runner.CreatePipeline(1, candidate_code, candidate.entry);
    const uint32_t effective_subgroup = runner.subgroup_size ? runner.subgroup_size : runner.default_subgroup;
    std::printf("GPU: %s (%s); subgroup %u%s; workgroup %ux%ux%u; dispatch %ux%ux%u; pipelines %.1f ms / %.1f ms\n",
                runner.device_name.c_str(), runner.driver_name.c_str(), effective_subgroup,
                runner.subgroup_size ? " (required)" : " (default)", baseline.workgroup[0], baseline.workgroup[1],
                baseline.workgroup[2], options.dispatch[0], options.dispatch[1], options.dispatch[2],
                runner.pipeline_ms[0], runner.pipeline_ms[1]);
    for (const auto& b : runner.buffers)
        std::printf("  set %u binding %u: %s x%u, %llu bytes, fill %s, compare %s\n", b.spec.set, b.spec.binding,
                    KindName(b.kind), b.count, static_cast<unsigned long long>(b.spec.size),
                    FillName(b.spec.fill), CompareName(b.spec.compare));

    std::vector<std::vector<uint8_t>> files(runner.buffers.size());
    for (size_t i = 0; i < runner.buffers.size(); ++i)
        if (runner.buffers[i].spec.fill == Fill::File) files[i] = ReadBytes(runner.buffers[i].spec.file);

    // Correctness: identical inputs per case, every compared buffer checked element-wise.
    uint64_t compared = 0, mismatches = 0, changed = 0, comparable = 0;
    double max_distance = 0;
    std::vector<Mismatch> shown;
    std::vector<std::vector<uint8_t>> inputs(runner.buffers.size()), before, after;
    for (uint32_t test = 0; test < options.cases; ++test) {
        for (size_t i = 0; i < runner.buffers.size(); ++i) {
            const auto& b = runner.buffers[i];
            inputs[i].clear();
            for (uint32_t element = 0; element < b.count; ++element) {
                const auto block = MakeInput(b.spec, test * 64 + element, files[i]);
                inputs[i].insert(inputs[i].end(), block.begin(), block.end());
            }
        }
        runner.Run(0, inputs, before, options.dispatch, 1);
        runner.Run(1, inputs, after, options.dispatch, 1);
        for (size_t i = 0; i < runner.buffers.size(); ++i) {
            const auto& spec = runner.buffers[i].spec;
            if (spec.compare == Compare::None) continue;
            const uint32_t element = spec.compare == Compare::UlpF16 ? 2 : 4;
            const uint64_t total = inputs[i].size();
            const uint64_t begin = std::min(spec.begin, total) / element * element;
            const uint64_t end = std::min(spec.end, total) / element * element;
            for (uint64_t offset = begin; offset < end; offset += element) {
                uint32_t a = 0, b = 0, in = 0;
                std::memcpy(&a, before[i].data() + offset, element);
                std::memcpy(&b, after[i].data() + offset, element);
                std::memcpy(&in, inputs[i].data() + offset, element);
                ++compared;
                if (test == 0) {
                    ++comparable;
                    changed += a != in;
                }
                if (const auto distance = Differ(spec.compare, a, b, spec.tolerance)) {
                    ++mismatches;
                    max_distance = std::max(max_distance, *distance);
                    if (shown.size() < options.show)
                        shown.push_back({test, spec.set, spec.binding, offset, a, b, *distance});
                }
            }
        }
    }
    if (comparable && changed == 0)
        std::fprintf(stderr, "WARNING: the baseline left every compared element equal to its input; "
                             "check the dispatch size and compared ranges.\n");
    for (const auto& m : shown)
        std::fprintf(stderr, "Mismatch: case=%u set=%u binding=%u byte=%llu baseline=%08x candidate=%08x distance=%g\n",
                     m.test, m.set, m.binding, static_cast<unsigned long long>(m.offset), m.baseline, m.candidate,
                     m.distance);
    if (mismatches)
        std::printf("FAIL: %llu of %llu compared elements differ over %u cases (max distance %g).\n",
                    static_cast<unsigned long long>(mismatches), static_cast<unsigned long long>(compared),
                    options.cases, max_distance);
    else
        std::printf("PASS: %llu elements compared over %u cases; %llu of %llu compared elements changed in case 0.\n",
                    static_cast<unsigned long long>(compared), options.cases,
                    static_cast<unsigned long long>(changed), static_cast<unsigned long long>(comparable));

    // Timing: alternate the order each sample and read GPU timestamps around the dispatch loop.
    std::array<Stats, 2> stats{};
    if (options.timing) {
        if (!runner.queries) throw std::runtime_error("Queue does not support GPU timestamps");
        std::vector<std::vector<uint8_t>> scratch;
        runner.Run(0, inputs, scratch, options.dispatch, std::min(options.timing, 16U));
        runner.Run(1, inputs, scratch, options.dispatch, std::min(options.timing, 16U));
        std::array<std::vector<double>, 2> samples;
        for (uint32_t sample = 0; sample < options.samples; ++sample)
            for (uint32_t order = 0; order < 2; ++order) {
                const uint32_t index = (sample + order) % 2;
                samples[index].push_back(runner.Run(index, inputs, scratch, options.dispatch, options.timing));
            }
        for (uint32_t i = 0; i < 2; ++i) stats[i] = Summarize(samples[i]);
        std::printf("GPU ns per dispatch (median of %u samples x %u serial dispatches): baseline %.1f "
                    "(min %.1f, p90 %.1f); candidate %.1f (min %.1f, p90 %.1f); ratio %.3f\n",
                    options.samples, options.timing, stats[0].median, stats[0].min, stats[0].p90,
                    stats[1].median, stats[1].min, stats[1].p90, stats[1].median / stats[0].median);
    }
    if (runner.validation_errors)
        std::printf("Validation reported %u errors.\n", runner.validation_errors);

    if (!options.json.empty()) {
        std::string out = "{\"device\":" + JsonString(runner.device_name) + ",\"driver\":" +
                          JsonString(runner.driver_name) + ",\"subgroup\":" + std::to_string(effective_subgroup) +
                          ",\"baseline\":" + ReflectionJson(baseline) + ",\"candidate\":" + ReflectionJson(candidate) +
                          ",\"pipeline_ms\":[" + JsonNumber(runner.pipeline_ms[0]) + "," + JsonNumber(runner.pipeline_ms[1]) +
                          "],\"cases\":" + std::to_string(options.cases) + ",\"compared\":" + std::to_string(compared) +
                          ",\"mismatches\":" + std::to_string(mismatches) + ",\"max_distance\":" + JsonNumber(max_distance) +
                          ",\"changed\":" + std::to_string(changed) + ",\"comparable\":" + std::to_string(comparable) +
                          ",\"validation_errors\":" + std::to_string(runner.validation_errors) + ",\"first_mismatches\":[";
        for (size_t i = 0; i < shown.size(); ++i) {
            const auto& m = shown[i];
            char buffer[256];
            std::snprintf(buffer, sizeof(buffer),
                          "%s{\"case\":%u,\"set\":%u,\"binding\":%u,\"byte\":%llu,\"baseline\":%u,\"candidate\":%u,\"distance\":%s}",
                          i ? "," : "", m.test, m.set, m.binding, static_cast<unsigned long long>(m.offset), m.baseline,
                          m.candidate, JsonNumber(m.distance).c_str());
            out += buffer;
        }
        out += "]";
        if (options.timing) {
            const auto one = [](const Stats& s) {
                return "{\"min\":" + JsonNumber(s.min) + ",\"median\":" + JsonNumber(s.median) + ",\"p90\":" + JsonNumber(s.p90) + "}";
            };
            out += ",\"timing\":{\"repeats\":" + std::to_string(options.timing) + ",\"samples\":" +
                   std::to_string(options.samples) + ",\"baseline\":" + one(stats[0]) + ",\"candidate\":" + one(stats[1]) +
                   ",\"ratio\":" + JsonNumber(stats[1].median / stats[0].median) + "}";
        }
        out += "}\n";
        std::ofstream file{options.json};
        file << out;
        if (!file) throw std::runtime_error("Cannot write " + options.json);
    }
    return mismatches || runner.validation_errors ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 2;
}
