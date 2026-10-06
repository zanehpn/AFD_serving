"""Load the shared stdlib-only layout contract without importing either BO package."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    'afd_native_layout_contract', Path(__file__).resolve().parents[1] /
    'bo_dse/scripts/afd/static_dse/native_layout.py')
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
FIELDS = _MODULE.FIELDS
validate = _MODULE.validate
deployment = _MODULE.deployment
replicas = _MODULE.replicas
enumerate_layouts = _MODULE.enumerate_layouts
verify_launch = _MODULE.verify_launch
