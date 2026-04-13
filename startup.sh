_startup_fail() {
  echo "$1" >&2
}

_startup_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_startup_venv_dir="${_startup_repo_root}/venv"
_startup_venv_activate="${_startup_venv_dir}/bin/activate"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  _startup_active_venv="$(cd "${VIRTUAL_ENV}" 2>/dev/null && pwd || true)"
  _startup_repo_venv="${_startup_venv_dir}"
  if [ -n "${_startup_active_venv}" ] && [ "${_startup_active_venv}" != "${_startup_repo_venv}" ]; then
    if type deactivate >/dev/null 2>&1; then
      echo "[startup] deactivating foreign virtualenv: ${VIRTUAL_ENV}" >&2
      deactivate
    fi
  fi
fi

if [ -f /sdf/group/hps/setup/hps-env.sh ]; then
  source /sdf/group/hps/setup/hps-env.sh || {
    _startup_fail "[startup] failed to source /sdf/group/hps/setup/hps-env.sh"
    return 1 2>/dev/null || exit 1
  }
fi

if [ -f /sdf/group/hps/sw2/conda/etc/profile.d/conda.sh ]; then
  source /sdf/group/hps/sw2/conda/etc/profile.d/conda.sh || {
    _startup_fail "[startup] failed to source the HPS conda profile"
    return 1 2>/dev/null || exit 1
  }
  conda activate hps || {
    _startup_fail "[startup] failed to activate the HPS conda environment"
    return 1 2>/dev/null || exit 1
  }
fi

if [ ! -f "${_startup_venv_activate}" ]; then
  _startup_fail "[startup] missing repo virtualenv at ${_startup_venv_dir}
[startup] bootstrap it with:
  cd ${_startup_repo_root}
  python3 -m venv venv
  source ${_startup_venv_activate}
  pip install -r requirements.txt
  pip install -e ."
  return 1 2>/dev/null || exit 1
fi

source "${_startup_venv_activate}" || {
  _startup_fail "[startup] failed to activate ${_startup_venv_activate}"
  return 1 2>/dev/null || exit 1
}
