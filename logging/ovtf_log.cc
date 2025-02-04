/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "ovtf_log.h"
#include <cstdlib>

using namespace std;

namespace {
// Parse log level (int64) from environment variable (char*)
tensorflow::int64 LogLevelStrToInt(const char* tf_env_var_val) {
  if (tf_env_var_val == nullptr) {
    return 0;
  }

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  string min_log_level(tf_env_var_val);
  std::istringstream ss(min_log_level);
  tensorflow::int64 level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0)
    level = 0;
  }

  return level;
}
}  // namespace

tensorflow::int64 NGraphLogMessage::MinNGraphVLogLevel() {
  const char* tf_env_var_val = std::getenv("OPENVINO_TF_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
}
