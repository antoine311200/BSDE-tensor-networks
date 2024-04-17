import importlib

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

reload_backend('numpy')  # or 'cupy'
