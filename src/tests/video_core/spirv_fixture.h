// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"

namespace TestFixtures {

/// Writes an emitted module to $EDEN_SHADER_FIXTURES/<name>.spv so tools/nx2/shader_ab.py can
/// run it on the GPU; a test that emits nothing when the variable is unset still checks its code.
inline void WriteSpirv(std::string_view name, const std::vector<u32>& code) {
    const char* directory = std::getenv("EDEN_SHADER_FIXTURES");
    if (!directory) {
        directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT");
    }
    if (!directory || name.empty()) {
        return;
    }
    std::filesystem::create_directories(directory);
    std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                       std::ios::binary};
    file.write(reinterpret_cast<const char*>(code.data()),
               static_cast<std::streamsize>(code.size() * sizeof(u32)));
    REQUIRE(file.good());
}

} // namespace TestFixtures
