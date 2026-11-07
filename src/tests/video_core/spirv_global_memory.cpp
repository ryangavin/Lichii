// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "spirv_fixture.h"

namespace {
using namespace Shader;

constexpr u32 NvnBufferBase = 0x110;

/// Each lane copies one element from the input half of storage buffer `lane % buffers` to the
/// output half of the next buffer, through the backend's global memory routing helpers. The
/// addresses are run-time values, so no pass can turn them into direct storage accesses.
struct GlobalMemoryProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Profile profile{};

    explicit GlobalMemoryProgram(u32 buffers, u32 width) {
        REQUIRE((buffers & (buffers - 1)) == 0);
        program.stage = Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        profile.supported_spirv = 0x00010500;
        profile.support_int64 = true;
        profile.support_descriptor_aliasing = true;
        profile.unified_descriptor_binding = true;
        profile.min_ssbo_alignment = 16;
        auto& info = program.info;
        info.uses_global_memory = true;
        info.uses_int64 = true;
        info.nvn_buffer_base = NvnBufferBase;
        info.uses_local_invocation_id = true;
        info.uses_workgroup_id = true;
        info.used_constant_buffer_types = IR::Type::U32 | IR::Type::U32x2;
        info.used_storage_buffer_types = IR::Type::U32 | IR::Type::U32x2 | IR::Type::U32x4;
        info.constant_buffer_descriptors = {{0, 1}};
        for (u32 i = 0; i < buffers; ++i) {
            info.storage_buffers_descriptors.push_back({0, NvnBufferBase + 16 * i, 1, true});
            info.nvn_buffer_used.set(i);
        }
        IR::IREmitter ir{block};
        const IR::U32 lane{ir.IAdd(ir.IMul(ir.WorkgroupIdX(), ir.Imm32(64)),
                                   ir.LocalInvocationIdX())};
        const IR::U32 source{ir.BitwiseAnd(lane, ir.Imm32(buffers - 1))};
        const IR::U32 destination{ir.BitwiseAnd(ir.IAdd(source, ir.Imm32(1)),
                                                ir.Imm32(buffers - 1))};
        const IR::U32 element{ir.IMul(ir.ShiftRightLogical(lane, ir.Imm32(std::countr_zero(buffers))),
                                      ir.Imm32(width))};
        const auto base = [&](const IR::U32& index) {
            const IR::U32 offset{ir.IAdd(ir.Imm32(NvnBufferBase),
                                         ir.ShiftLeftLogical(index, ir.Imm32(4)))};
            const IR::U32 low{ir.GetCbuf(ir.Imm32(0), offset)};
            const IR::U32 high{ir.GetCbuf(ir.Imm32(0), ir.IAdd(offset, ir.Imm32(4)))};
            return ir.PackUint2x32(ir.CompositeConstruct(low, high));
        };
        const IR::U64 read{ir.IAdd(base(source), ir.UConvert(64, ir.IAdd(element, ir.Imm32(32768))))};
        const IR::U64 write{ir.IAdd(base(destination), ir.UConvert(64, element))};
        switch (width) {
        case 4: ir.WriteGlobal32(write, ir.LoadGlobal32(read)); break;
        case 8: ir.WriteGlobal64(write, ir.LoadGlobal64(read)); break;
        case 16: ir.WriteGlobal128(write, ir.LoadGlobal128(read)); break;
        default: FAIL("Unsupported width");
        }
    }

    std::vector<u32> Emit(const std::string& name) {
        auto code = Backend::SPIRV::EmitSPIRV(profile, program);
        TestFixtures::WriteSpirv(name, code);
        return code;
    }
};

size_t Count(const std::vector<u32>& code, u32 opcode) {
    size_t result = 0;
    for (size_t offset = 5; offset < code.size();) {
        const u32 size = code[offset] >> 16;
        REQUIRE(size != 0);
        result += (code[offset] & 0xffffU) == opcode;
        offset += size;
    }
    return result;
}
} // namespace

TEST_CASE("Global memory routing fixtures cover one to eight NVN buffers",
          "[video_core][global-memory][fixture]") {
    for (const u32 buffers : {1U, 4U, 8U}) {
        GlobalMemoryProgram fixture{buffers, 4};
        const auto code = fixture.Emit("global-u32-" + std::to_string(buffers));
        // Load and store helpers each branch once per candidate buffer.
        REQUIRE(Count(code, 250) >= 2 * buffers); // OpBranchConditional
        REQUIRE(Count(code, 57) == 2);            // OpFunctionCall: one load, one store
    }
    for (const u32 width : {8U, 16U}) {
        GlobalMemoryProgram fixture{4, width};
        const auto code = fixture.Emit("global-u32x" + std::to_string(width / 4) + "-4");
        REQUIRE(Count(code, 57) == 2);
    }
}
