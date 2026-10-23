import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PlatformWheelBuildHook(BuildHookInterface):
    """
    Sets the wheel's platform tag from the PYNFFT3_WHEEL_TAG environment
    variable, so CI can build one platform-specific wheel per (OS, arch,
    glibc floor) combination from the same source tree - each build already
    has only the relevant subset of src/pyNFFT3/lib/ present (removed by the
    CI step that sets this variable; see release.yml), this hook just makes
    the wheel's filename/tag honest about that. Without the variable (e.g. a
    local `pip install .`), this is a no-op and hatchling picks its own
    default tag, as before.
    """

    PLUGIN_NAME = "pynfft3-platform-tag"

    def initialize(self, version, build_data):
        tag = os.environ.get("PYNFFT3_WHEEL_TAG")
        if tag:
            build_data["tag"] = tag
            build_data["pure_python"] = False
