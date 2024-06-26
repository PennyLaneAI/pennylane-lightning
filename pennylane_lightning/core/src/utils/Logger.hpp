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

#include <spdlog/spdlog.h>

#define LOGGER_INFO(...) \
    { spdlog::info("[{1}][Line: {2}][Function: {3}({0})]", __VA_ARGS__, __FILE__, __LINE__, __func__); }

#define LOGGER_DEBUG(...) \
    { spdlog::debug("[{1}][Line: {2}][Function: {3}({0})]", __VA_ARGS__, __FILE__, __LINE__, __func__); }

#define LOGGER_WARN(...) \
    { spdlog::warn("[{1}][Line: {2}][Function: {3}({0})]", __VA_ARGS__, __FILE__, __LINE__, __func__); }

#define LOGGER_ERROR(...) \
    { spdlog::error("[{1}][Line: {2}][Function: {3}({0})]", __VA_ARGS__, __FILE__, __LINE__, __func__); }

#define LOGGER_CRITICAL(...) \
    { spdlog::critical("[{1}][Line: {2}][Function: {3}({0})]", __VA_ARGS__, __FILE__, __LINE__, __func__); }

#define LOGGER_TRACE(...) \
    { spdlog::trace("{3}({0}) [{1}:{2}]", __VA_ARGS__, __FILE__, __LINE__, __func__); }
