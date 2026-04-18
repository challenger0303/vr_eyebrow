"""PyInstaller runtime hook: resolve DLL search paths for onnxruntime and PyQt5.

Critical fix: some systems have an older/ABI-incompatible onnxruntime.dll in
C:\\Windows\\System32 (installed by Windows Copilot, Windows ML, Visual Studio,
or similar). Since System32 is searched before the app directory, the pyd
would bind to the wrong DLL and fail with "DLL initialization routine failed".
We pre-load our bundled onnxruntime.dll via ctypes so subsequent imports
resolve to our version.
"""
import os
import sys
import shutil
import ctypes

if hasattr(sys, '_MEIPASS'):
    qt_bin = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt5', 'bin')
    if os.path.isdir(qt_bin):
        try:
            os.add_dll_directory(qt_bin)
        except (OSError, AttributeError):
            pass
        os.environ['PATH'] = qt_bin + os.pathsep + os.environ.get('PATH', '')

    qt_plugins = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt5', 'plugins')
    if os.path.isdir(qt_plugins):
        os.environ['QT_PLUGIN_PATH'] = qt_plugins
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_plugins, 'platforms')

    ort_capi = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
    if os.path.isdir(ort_capi):
        # Prepend our path so LoadLibrary finds our DLLs before System32
        try:
            os.add_dll_directory(ort_capi)
        except (OSError, AttributeError):
            pass
        os.environ['PATH'] = ort_capi + os.pathsep + os.environ.get('PATH', '')

        # Copy DLLs to _MEIPASS root as a second fallback
        for dll in ('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll'):
            src = os.path.join(ort_capi, dll)
            dst = os.path.join(sys._MEIPASS, dll)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

        # CRITICAL: Pre-load our bundled DLLs explicitly to beat System32 version.
        # Load order matters: DirectML first, then onnxruntime_providers_shared,
        # then onnxruntime. By the time _pybind_state.pyd is imported, the
        # correct onnxruntime.dll is already in the process's loaded modules
        # and will be used for dependency resolution instead of System32's copy.
        for dll in ('DirectML.dll', 'onnxruntime_providers_shared.dll', 'onnxruntime.dll'):
            path = os.path.join(ort_capi, dll)
            if os.path.exists(path):
                try:
                    ctypes.WinDLL(path)
                except OSError:
                    # Fallback: try from _MEIPASS root
                    alt = os.path.join(sys._MEIPASS, dll)
                    if os.path.exists(alt):
                        try:
                            ctypes.WinDLL(alt)
                        except OSError:
                            pass
