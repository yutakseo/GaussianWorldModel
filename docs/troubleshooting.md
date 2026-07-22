# Troubleshooting

- ImportError: .../.venv/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12

Check [[tool.uv.index]] and [tool.uv.sources.torch], make sure the cuda version is correct.

- ImportError: cannot import name 'packaging' from 'pkg_resources'
```bash
pip install setuptools==69.5.1
```

- ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```bash
pip install numpy==1.26.4
```

- _pickle.UnpicklingError: invalid load key, 'v'.
use `git lfs install`
