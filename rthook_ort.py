"""PyInstaller runtime hook: resolve DLL search paths for onnxruntime and PyQt5."""
import os
import sys
import shutil

if hasattr(sys, '_MEIPASS'):
    # Add PyQt5 Qt5 bin directory to DLL search path (for Qt5Widgets.dll etc.)
    qt_bin = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt5', 'bin')
    if os.path.isdir(qt_bin):
        try:
            os.add_dll_directory(qt_bin)
        except (OSError, AttributeError):
            pass
        os.environ['PATH'] = qt_bin + os.pathsep + os.environ.get('PATH', '')

    # Set Qt plugin path so platforms/qwindows.dll is found
    qt_plugins = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt5', 'plugins')
    if os.path.isdir(qt_plugins):
        os.environ['QT_PLUGIN_PATH'] = qt_plugins
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_plugins, 'platforms')

    ort_capi = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
    if os.path.isdir(ort_capi):
        try:
            os.add_dll_directory(ort_capi)
        except (OSError, AttributeError):
            pass
        os.environ['PATH'] = ort_capi + os.pathsep + os.environ.get('PATH', '')

        for dll in ('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll'):
            src = os.path.join(ort_capi, dll)
            dst = os.path.join(sys._MEIPASS, dll)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
