// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Logger macros using spdlog functions
 */
#pragma once

#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#include <spdlog/spdlog.h>

#define LOGGER_INFO(...)                                                       \
    { spdlog::info(__VA_ARGS__); }

#define LOGGER_DEBUG(...)                                                      \
    {                                                                          \
        spdlog::debug("[{0}:{1}] Function: {2}({3})", __FILE__, __LINE__,      \
                      __func__, __VA_ARGS__);                                  \
    }

#define LOGGER_WARN(...)                                                       \
    { spdlog::warn(__VA_ARGS__); }

#define LOGGER_TRACE(...)                                                      \
    {                                                                          \
        spdlog::debug("[{0}:{1}] Function: {2}({3})", __FILE__, __LINE__,      \
                      __func__, __VA_ARGS__);                                  \
    }

static inline void set_logger_level_from_env() {
    const char *env_log_level = std::getenv("LOGGER_LEVEL");
    if (env_log_level != nullptr) {
        std::string level_str(env_log_level);
        if (level_str == "info") {
            spdlog::set_level(spdlog::level::info);
        } else if (level_str == "debug") {
            spdlog::set_level(spdlog::level::debug);
        } else if (level_str == "warn") {
            spdlog::set_level(spdlog::level::warn);
        } else if (level_str == "trace") {
            spdlog::set_level(spdlog::level::trace);
        } else {
            LOGGER_WARN(
                "Invalid level set in LOGGER_LEVEL; "
                "supported levels are 'info', 'debug', 'warn', and 'trace'");
        }
    } else {
        spdlog::set_level(spdlog::level::off);
    }
}
