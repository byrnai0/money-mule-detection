___

Ok so in phase 3, we need to install torch package, but its only for python v3.11
so download that first, do all the environmental variable shit and then go back deactivate the current venv (it was v3.13) and create another venv specifically from v3.11

```bash
deactivate
PS C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection> Remove-Item -Recurse -Force .venv
PS C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection> py -3.11 -m venv .venv
PS C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection> .\.venv\Scripts\Activate.ps1
(.venv) PS C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection> python --version
Python 3.11.9
(.venv) PS C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection> python -c "import sys; print(sys.executable)"
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\.venv\Scripts\python.exe
(.venv) PS C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection> python -m pip --version
pip 24.0 from C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\.venv\Lib\site-packages\pip (python 3.11)
```

after all these shit, we will get `ModuleNotFoundError: No module named 'torch'`

## 1. First verify the environment

Make sure you're in:

`C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection`

and your `.venv` is activated:

`.\.venv\Scripts\Activate.ps1`

Then run:

```
python --version
python -c "import sys; print(sys.executable)"
```

You should get:

```
Python 3.11.x
C:\Users\csp\OneDrive\Documents\VSCode\money_mule_detection\.venv\Scripts\python.exe
```

## 2. Check your NVIDIA driver

Run:
`nvidia-smi`

Don't run:

`pip install torch`

and don't install CUDA Toolkit separately just because PyTorch mentions CUDA. PyTorch's pip packages bundle the required CUDA runtime components; your NVIDIA driver is the important system-side prerequisite for using the GPU.

Once you paste the `nvidia-smi` output, I'll give you the exact command for:

```
PyTorch

→ CUDA acceleration

→ PyTorch Geometric

→ graph construction test
```

Because we're setting this environment up in August 2026, lets use:

```
Python     3.11

PyTorch    2.12.1

CUDA       13.2 PyTorch wheel

PyG        current compatible release
```

```
### 1. Make sure your venv is active

From your project root:

.\.venv\Scripts\Activate.ps1

Verify:

python --version

You want:

Python 3.11.x

Then verify:

python -m pip --version

It should point inside:

money_mule_detection\.venv\

---

## 2. Install PyTorch

Run exactly:

python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132

That is an official PyTorch-provided CUDA 13.2 wheel combination.

You **don't need to install CUDA Toolkit separately** for this.

---

## 3. Immediately test PyTorch

Run:

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

We want something approximately like:

PyTorch: 2.12.1

CUDA build: 13.2

CUDA available: True

GPU: NVIDIA GeForce RTX 4060

**Do not proceed if `CUDA available` says `False`.** Send me that output and we'll fix it first.

---

# 4. Then install PyTorch Geometric

Once PyTorch works:

python -m pip install torch-geometric

Current PyG documentation says that from PyG 2.3 onward, the basic installation only requires PyTorch; the additional packages such as `pyg_lib`, `torch_scatter`, and `torch_sparse` are optional unless we actually need those features.

So we're **not** going to repeat my earlier recommendation of manually installing every PyG dependency. That was unnecessary.

---

## 5. Test PyG

Run:

python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA:', torch.cuda.is_available())"

Then:

python -c "from torch_geometric.data import Data; print('PyG Data import: OK')"

---

# 6. Only after those tests

Then run our graph builder:

python -u .\src\graph\builder.py

At that point Phase 3 should finally be able to start.
```
