#!/usr/bin/env bash

# Source this file from the repo root:
#   source env.sh
#
# It configures the environment expected by the 3D Light-Dark scripts
# without touching your global shell config.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed."
  echo "Use: source env.sh"
  exit 1
fi

VTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${HUMANAV_ROOT:-}" ]]; then
  HUMANAV_ROOT="$(cd "${HUMANAV_ROOT}" && pwd)"
elif [[ -d "${VTS_ROOT}/../HumANav-Release" ]]; then
  HUMANAV_ROOT="$(cd "${VTS_ROOT}/../HumANav-Release" && pwd)"
elif [[ -d "${VTS_ROOT}/../HumANavRelease" ]]; then
  HUMANAV_ROOT="$(cd "${VTS_ROOT}/../HumANavRelease" && pwd)"
else
  echo "Could not find HumANav-Release next to this repo."
  echo "Set HUMANAV_ROOT first, then run: source env.sh"
  return 1
fi

case ":${PYTHONPATH:-}:" in
  *":${VTS_ROOT}:"*) ;;
  *) export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${VTS_ROOT}" ;;
esac

case ":${PYTHONPATH:-}:" in
  *":${HUMANAV_ROOT}:"*) ;;
  *) export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${HUMANAV_ROOT}" ;;
esac

export PYTHONPATH=/home/himanshu/Documents/Research/VBA/external:$PYTHONPATH

export PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

echo "Configured VisualTreeSearch environment"
echo "VTS_ROOT=${VTS_ROOT}"
echo "HUMANAV_ROOT=${HUMANAV_ROOT}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
echo "LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE}"
echo "MESA_LOADER_DRIVER_OVERRIDE=${MESA_LOADER_DRIVER_OVERRIDE}"
