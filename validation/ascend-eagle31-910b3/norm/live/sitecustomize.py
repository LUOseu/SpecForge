"""Install the optional probe after SGLang imports its genuine draft module."""

import importlib.machinery
import os
import sys


class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'sglang.srt.models.llama_eagle3':
            return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        original=spec.loader.exec_module
        def observed(module):
            original(module)
            from eagle31_norm_observer import install
            install(module)
        spec.loader.exec_module=observed
        return spec


if os.environ.get('E31_NORM_LOGDIR'):
    sys.meta_path.insert(0,Finder())
