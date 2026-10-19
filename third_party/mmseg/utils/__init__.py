# Adapted minimal dependency surface; original OpenMMLab license retained.
from .typing_utils import *
from .misc import add_prefix, stack_batch
from .io import datafrombytes

def register_all_modules(init_default_scope=True):
    import mmseg.models
    import mmseg.datasets
    if init_default_scope:
        from mmengine.registry import init_default_scope as initialize
        initialize('mmseg')
