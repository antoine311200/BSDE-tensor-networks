import importlib
import sys

def reload_backend(backend: str):
    global xp
    if backend == 'numpy':
        module = 'numpy'
    elif backend == 'cupy':
        module = 'cupy'
    else:
        raise ValueError(f"Unknown backend {backend}")
    
    try:
        xp = importlib.import_module(module)
    except ImportError:
        raise ImportError(f"Failed to import backend module: {module}")
    
def update_imports(module_name):
    for module in list(sys.modules.keys()):
        if module.startswith('bsde_solver') and module != 'bsde_solver':
            try:
                # Reload the module to update the imports
                imported_module = importlib.import_module(module)
                if hasattr(imported_module, 'xp'):
                    imported_module.xp = importlib.import_module(module_name)
            except ImportError:
                print(f"Failed to update imports for module: {module}")

reload_backend('numpy')  # or 'cupy'
