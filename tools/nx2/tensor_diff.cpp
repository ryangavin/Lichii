// SPDX-License-Identifier: GPL-3.0-or-later
// Build: c++ -std=c++20 -O2 tools/nx2/tensor_diff.cpp -lvulkan -o /tmp/tensor-diff
// Emit fixtures: EDEN_TENSOR_TEST_OUTPUT=/tmp/before build/dev/bin/tests '[tensor]'
// Run: /tmp/tensor-diff before/software-k16-inputs.spv after/software-k16-inputs.spv
// The *-inputs fixtures read eight packed fragment words per invocation after 512 output bytes.

#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t OutputWords = 128;
constexpr uint32_t BufferWords = OutputWords + 64 * 8;
constexpr VkDeviceSize BufferBytes = BufferWords * sizeof(uint32_t);

void Check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string{operation} + ": VkResult=" + std::to_string(result));
}

std::vector<uint32_t> ReadShader(const char* path) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file || file.tellg() < 20 || file.tellg() % 4 != 0)
        throw std::runtime_error(std::string{"Invalid SPIR-V file: "} + path);
    std::vector<uint32_t> code(static_cast<size_t>(file.tellg()) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), code.size() * 4);
    if (!file || code[0] != 0x07230203)
        throw std::runtime_error(std::string{"Invalid SPIR-V header: "} + path);
    return code;
}

struct Runner {
    VkInstance instance{};
    VkDevice device{};
    VkQueue queue{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
    VkDescriptorSetLayout set_layout{};
    VkDescriptorPool descriptor_pool{};
    VkDescriptorSet descriptor{};
    VkPipelineLayout layout{};
    VkCommandPool command_pool{};
    VkCommandBuffer command{};
    VkFence fence{};
    VkQueryPool queries{};
    VkPipeline pipelines[2]{};
    uint32_t subgroup_size{};
    uint32_t timestamp_bits{};
    float timestamp_period{};

    ~Runner() {
        if (device) {
            vkDeviceWaitIdle(device);
            for (auto pipeline : pipelines) if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (queries) vkDestroyQueryPool(device, queries, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
            if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            if (mapped) vkUnmapMemory(device, memory);
            if (buffer) vkDestroyBuffer(device, buffer, nullptr);
            if (memory) vkFreeMemory(device, memory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    void Init(uint32_t device_index, uint32_t requested_subgroup) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Lichii tensor differential test";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create.pApplicationInfo = &app;
        Check(vkCreateInstance(&create, nullptr, &instance), "vkCreateInstance");
        uint32_t count{};
        Check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
        std::vector<VkPhysicalDevice> devices(count);
        Check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
        if (device_index >= count) throw std::runtime_error("Requested GPU index does not exist");
        const auto physical = devices[device_index];
        VkPhysicalDeviceSubgroupSizeControlProperties size_props{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        subgroup.pNext = &size_props;
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physical, &props);
        if (props.properties.apiVersion < VK_API_VERSION_1_3)
            throw std::runtime_error("This standalone runner requires Vulkan 1.3");
        if (!(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
            !(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT))
            throw std::runtime_error("GPU lacks compute subgroup shuffle support");
        VkPhysicalDeviceSubgroupSizeControlFeatures sizes{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &sizes;
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (requested_subgroup && (!sizes.subgroupSizeControl ||
            !(size_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
            requested_subgroup < size_props.minSubgroupSize ||
            requested_subgroup > size_props.maxSubgroupSize))
            throw std::runtime_error("Requested compute subgroup size is unsupported");
        subgroup_size = requested_subgroup;
        const auto effective_size = subgroup_size ? subgroup_size : subgroup.subgroupSize;
        if (effective_size != 32 && effective_size != 64)
            throw std::runtime_error("Tensor fixture requires a 32- or 64-lane subgroup");
        std::printf("GPU: %s; subgroup: %u%s\n", props.properties.deviceName,
                    effective_size, subgroup_size ? " (required)" : " (default)");
        timestamp_period = props.properties.limits.timestampPeriod;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
        uint32_t family = 0;
        while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
        if (family == count) throw std::runtime_error("No compute queue");
        timestamp_bits = families[family].timestampValidBits;
        const float priority = 1;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        sizes.subgroupSizeControl = requested_subgroup != 0;
        sizes.computeFullSubgroups = VK_FALSE;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.pNext = &sizes;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        Check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, family, 0, &queue);
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = BufferBytes;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memories;
        vkGetPhysicalDeviceMemoryProperties(physical, &memories);
        uint32_t memory_type = 0;
        const auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        while (memory_type < memories.memoryTypeCount &&
               (!(requirements.memoryTypeBits & (1U << memory_type)) ||
                (memories.memoryTypes[memory_type].propertyFlags & flags) != flags)) ++memory_type;
        if (memory_type == memories.memoryTypeCount)
            throw std::runtime_error("No host-visible coherent storage-buffer memory");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type;
        Check(vkAllocateMemory(device, &allocation, nullptr, &memory), "vkAllocateMemory");
        Check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        Check(vkMapMemory(device, memory, 0, BufferBytes, 0, &mapped), "vkMapMemory");
        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                  VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 1;
        set_info.pBindings = &binding;
        Check(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout), "create set layout");
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set_layout;
        Check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout), "create pipeline layout");
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        Check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "create descriptor pool");
        VkDescriptorSetAllocateInfo set_allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_allocation.descriptorPool = descriptor_pool;
        set_allocation.descriptorSetCount = 1;
        set_allocation.pSetLayouts = &set_layout;
        Check(vkAllocateDescriptorSets(device, &set_allocation, &descriptor), "allocate descriptor set");
        const VkDescriptorBufferInfo buffer_binding{buffer, 0, BufferBytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptor;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_binding;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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

    void Pipeline(uint32_t index, const std::vector<uint32_t>& code) {
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
        pipeline.stage.pName = "main";
        pipeline.layout = layout;
        const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline,
                                                     nullptr, &pipelines[index]);
        vkDestroyShaderModule(device, module, nullptr);
        Check(result, "create compute pipeline");
    }

    double Run(uint32_t index, const std::array<uint32_t, BufferWords>& data,
               std::array<uint32_t, OutputWords>& output, uint32_t repeats = 1) {
        std::memcpy(mapped, data.data(), BufferBytes);
        Check(vkResetCommandBuffer(command, 0), "reset command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(vkBeginCommandBuffer(command, &begin), "begin command buffer");
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[index]);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                                &descriptor, 0, nullptr);
        if (queries) {
            vkCmdResetQueryPool(command, queries, 0, 2);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
        }
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            vkCmdDispatch(command, 1, 1, 1);
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            if (repeat + 1 < repeats)
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                                     0, nullptr, 0, nullptr);
        }
        if (queries) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        Check(vkEndCommandBuffer(command), "end command buffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        Check(vkQueueSubmit(queue, 1, &submit, fence), "submit compute");
        Check(vkWaitForFences(device, 1, &fence, VK_TRUE, 30'000'000'000ULL), "wait for compute");
        Check(vkResetFences(device, 1, &fence), "reset fence");
        std::memcpy(output.data(), mapped, output.size() * 4);
        if (!queries) return 0;
        std::array<uint64_t, 2> times;
        Check(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(times), times.data(), sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "read timestamps");
        const uint64_t mask = timestamp_bits == 64 ? UINT64_MAX : (1ULL << timestamp_bits) - 1;
        return static_cast<double>((times[1] - times[0]) & mask) * timestamp_period / repeats;
    }
};

std::array<uint32_t, BufferWords> Inputs(uint32_t test) {
    std::array<uint32_t, BufferWords> data;
    std::fill(data.begin(), data.begin() + OutputWords, 0xcdcdcdcdU);
    std::mt19937 random{0x4d4d4100U + test};
    constexpr std::array<uint16_t, 20> edges{0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff,
        0x0400, 0x8400, 0x3555, 0xb555, 0x3bff, 0x3c00, 0x3c01, 0xbc00,
        0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe01};
    const auto half = [&](uint32_t position) -> uint32_t {
        switch (test % 8) {
        case 0: return edges[(position + test / 8) % edges.size()];
        case 1: return random() & 0xffffU;
        case 2: return (random() & 0x83ffU) | ((8U + random() % 9U) << 10);
        default: return random() & 0x87ffU;
        }
    };
    for (uint32_t i = OutputWords; i < BufferWords; ++i) {
        const auto low = half((i - OutputWords) * 2);
        data[i] = low | (half((i - OutputWords) * 2 + 1) << 16);
    }
    if (test % 8 >= 4) {
        // Isolate k=0 so C + A*B lands at a half-way point, subnormal, or overflow boundary.
        constexpr std::array<uint16_t, 4> a{0x1000, 0x0001, 0x0001, 0x4c00};
        constexpr std::array<uint16_t, 4> b{0x3c00, 0x3800, 0x3c00, 0x3c00};
        constexpr std::array<uint16_t, 4> c{0x3c00, 0x0000, 0x03ff, 0x7bff};
        const uint32_t pattern = test % 8 - 4;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t base = OutputWords + lane * 8;
            std::fill(data.begin() + base, data.begin() + base + 8, 0);
            if (lane % 4 == 0) {
                data[base] = data[base + 1] = a[pattern];
                data[base + 4] = b[pattern];
            }
            const uint32_t sign = ((test / 8) & 1) << 15;
            const uint32_t accumulator = (c[pattern] + ((test / 16) & 1)) | sign;
            data[base + 6] = data[base + 7] = accumulator | (accumulator << 16);
        }
    }
    return data;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s BASELINE.spv CANDIDATE.spv [--cases N] [--device N] "
                             "[--subgroup 32|64] [--timing N]\n", argv[0]);
        return 2;
    }
    uint32_t cases = 256, device = 0, subgroup = 0, timing = 0;
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("Missing option value");
        const auto value = std::stoul(argv[i + 1]);
        if (value > 100000) throw std::runtime_error("Option value exceeds 100000");
        const std::string option = argv[i];
        if (option == "--cases") cases = value;
        else if (option == "--device") device = value;
        else if (option == "--subgroup" && (value == 32 || value == 64)) subgroup = value;
        else if (option == "--timing") timing = value;
        else throw std::runtime_error("Unknown option or invalid subgroup size: " + option);
    }
    if (!cases) throw std::runtime_error("At least one comparison case is required");
    Runner runner;
    runner.Init(device, subgroup);
    runner.Pipeline(0, ReadShader(argv[1]));
    runner.Pipeline(1, ReadShader(argv[2]));
    uint64_t compared = 0;
    for (uint32_t test = 0; test < cases; ++test) {
        const auto data = Inputs(test);
        std::array<uint32_t, OutputWords> before, after;
        runner.Run(0, data, before);
        runner.Run(1, data, after);
        for (uint32_t word = 0; word < OutputWords; ++word) {
            if (before[word] == 0xcdcdcdcdU || after[word] == 0xcdcdcdcdU)
                throw std::runtime_error("Fixture left an output word unwritten");
            if (before[word] != after[word]) {
                std::fprintf(stderr, "Mismatch: case=%u seed=%08x lane=%u word=%u "
                                     "baseline=%08x candidate=%08x\n", test,
                             0x4d4d4100U + test, word / 2, word % 2, before[word], after[word]);
                return 1;
            }
            ++compared;
        }
    }
    std::printf("PASS: %u cases, %llu packed output words compared bitwise.\n", cases,
                static_cast<unsigned long long>(compared));
    if (timing) {
        if (!runner.queries) throw std::runtime_error("Queue does not support GPU timestamps");
        const auto data = Inputs(2);
        std::array<uint32_t, OutputWords> output;
        runner.Run(0, data, output, 16);
        runner.Run(1, data, output, 16);
        std::array<std::vector<double>, 2> samples;
        for (uint32_t sample = 0; sample < 10; ++sample)
            for (uint32_t order = 0; order < 2; ++order) {
                const uint32_t index = (sample + order) % 2;
                samples[index].push_back(runner.Run(index, data, output, timing));
            }
        for (auto& values : samples) std::sort(values.begin(), values.end());
        std::printf("GPU timestamp median per serial dispatch (includes barriers): "
                    "baseline=%.1f ns candidate=%.1f ns; %u repeats/sample.\n",
                    (samples[0][4] + samples[0][5]) / 2,
                    (samples[1][4] + samples[1][5]) / 2, timing);
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 2;
}
