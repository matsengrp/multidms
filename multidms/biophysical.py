"""
The multidms.biophysical module has been deprecated in version 2.0.0.

All modeling functionality now uses the jaxmodels backend internally.
The user-facing API (Data, Model, ModelCollection) remains compatible
with v1.x code, but the biophysical module is no longer available.

If you need the legacy biophysical module, please use multidms v1.x:
    pip install "multidms<2.0.0"

For migration guidance, see the documentation at:
    https://multidms.readthedocs.io/en/latest/migration_v2.html
"""

raise ImportError(
    "The multidms.biophysical module has been deprecated in version 2.0.0. "
    "All modeling functionality now uses the jaxmodels backend. "
    "Please see the migration guide at https://multidms.readthedocs.io/en/latest/migration_v2.html "
    "or use multidms v1.x if you require the legacy biophysical module."
)
