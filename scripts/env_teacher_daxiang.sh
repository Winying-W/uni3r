#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env_daxiang.sh"
export UNI3R_TEACHER_ENV="$UNI3R_EFS_ROOT/envs/teacher"
export PATH="$UNI3R_TEACHER_ENV/bin:$PATH"
export MASKCLUSTERING_ROOT="$UNI3R_ROOT/external/MaskClustering"
