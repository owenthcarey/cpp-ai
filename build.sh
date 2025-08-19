#!/usr/bin/env bash
set -euo pipefail

# Simple configure, build, and run script for this repo.
# Usage: ./build.sh [Release|Debug] [--with-dl]
#   default build type: Release

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="Release"
WITH_DL="OFF"

# Parse args
for arg in "$@"; do
	case "$arg" in
		Release|Debug)
			BUILD_TYPE="$arg"
			;;
		--with-dl)
			WITH_DL="ON"
			;;
		*)
			echo "Unknown argument: $arg" >&2
			echo "Usage: ./build.sh [Release|Debug] [--with-dl]" >&2
			exit 2
			;;
	esac
done

# Determine parallel build jobs (macOS)
JOBS_FLAG=""
if command -v sysctl >/dev/null 2>&1; then
	CORES="$(sysctl -n hw.ncpu || echo 4)"
	JOBS_FLAG="-j ${CORES}"
fi

# Prefer VCPKG_ROOT if provided to override toolchain path
TOOLCHAIN_ARG=""
if [[ -n "${VCPKG_ROOT:-}" && -f "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" ]]; then
	TOOLCHAIN_ARG="-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
fi

mkdir -p "${BUILD_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCPP_AI_BUILD_DL="${WITH_DL}" ${TOOLCHAIN_ARG}
cmake --build "${BUILD_DIR}" ${JOBS_FLAG}

exec "${BUILD_DIR}/cpp_ai"
