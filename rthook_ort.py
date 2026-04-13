"""PyInstaller runtime hook: copy onnxruntime DLLs to where they can be found."""
import os
import sys
import shutil

if hasattr(sys, '_MEIPASS'):
    ort_capi = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
    if os.path.isdir(ort_capi):
        # Add to DLL search path
        try:
            os.add_dll_directory(ort_capi)
        except (OSError, AttributeError):
            pass
        os.environ['PATH'] = ort_capi + os.pathsep + os.environ.get('PATH', '')

        # Also copy critical DLLs to _MEIPASS root as fallback
        for dll in ('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll'):
            src = os.path.join(ort_capi, dll)
            dst = os.path.join(sys._MEIPASS, dll)
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
