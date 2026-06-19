While I cannot directly package and send you a downloadable `.zip` file through this text interface, I have provided the **exact file structure and 100% complete source code** below. You can copy and paste these files into a folder on your computer and compress it into a `.zip` file yourself in less than two minutes.

This project demonstrates a high-performance data processing pipeline combining **Clean Architecture**, **Asyncio**, a **CPython C Extension** that releases the GIL, and an integrated **Profiling script**.

---

## 📂 Project File Structure

Create a directory named `advanced_python_project` with the following structure:

```text
advanced_python_project/
│
├── c_extensions/
│   └── fast_math.c
│
├── app/
│   ├── __init__.py
│   ├── domain.py
│   ├── use_cases.py
│   └── infrastructure.py
│
├── setup.py
├── main.py
└── TUTORIAL.md

```

---

## 🛠️ Codebase Implementation

### 1. The CPython C Extension

Save this as `c_extensions/fast_math.c`. It contains heavy math processing and safely handles releasing the Global Interpreter Lock (GIL).

```c
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// The core computational function
static PyObject* fast_sum_of_squares(PyObject* self, PyObject* args) {
    long iterations;

    // Parse incoming Python arguments (expects a single python long integer)
    if (!PyArg_ParseTuple(args, "l", &iterations)) {
        return NULL; // Raises TypeError automatically
    }

    long long total_sum = 0;

    // RELEASE THE GIL: Safe because we aren't touching any Python objects in this loop
    Py_BEGIN_ALLOW_THREADS

    for (long i = 0; i < iterations; i++) {
        total_sum += (i * i);
    }

    // RE-ACQUIRE THE GIL: Must happen before creating the Python return object
    Py_END_ALLOW_THREADS

    // Returns a "New Reference" to a Python long object
    return PyLong_FromLongLong(total_sum);
}

// Method definition structure
static PyMethodDef FastMathMethods[] = {
    {"sum_of_squares", fast_sum_of_squares, METH_VARARGS, "Compute sum of squares efficiently while releasing the GIL."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition structure
static struct PyModuleDef fastmathmodule = {
    PyModuleDef_HEAD_INIT,
    "fast_math", 
    "High-performance C extension for CPython",
    -1, 
    FastMathMethods
};

// Module initialization function called by Python during import
PyMODINIT_FUNC PyInit_fast_math(void) {
    return PyModuleCreate(&fastmathmodule);
}

```

### 2. C Extension Compilation Script

Save this as `setup.py` in the root directory.

```python
from setuptools import setup, Extension

def main():
    setup(
        name="fast_math",
        version="1.0.0",
        description="Python C Extension for heavy processing",
        ext_modules=[
            Extension("fast_math", ["c_extensions/fast_math.c"])
        ],
    )

if __name__ == "__main__":
    main()

```

### 3. Clean Architecture: Domain Layer

Save this as `app/domain.py`. It uses structural subtyping (`Protocol`) to build abstractions completely independent of external dependencies or frameworks.

```python
from typing import Protocol, List
from dataclasses import dataclass

@dataclass(frozen=True)
def JobResult:
    job_id: str
    iterations: int
    calculated_sum: int
    execution_time_ms: float

# The Structural Interface (Dependency Inversion principle)
class CalculationRepository(Protocol):
    async def fetch_pending_jobs(self) -> List[int]:
        ...

    async def save_results(self, results: List[JobResult]) -> None:
        ...

```

### 4. Clean Architecture: Use Cases Layer

Save this as `app/use_cases.py`. This contains purely asynchronous core business logic orchestration.

```python
import asyncio
import time
import fast_math  # Imported compiled C extension
from typing import List
from app.domain import CalculationRepository, JobResult

class BatchProcessingUseCase:
    """
    Coordinates data fetching, schedules high-performance processing across 
    worker pools, and saves results back to storage via abstractions.
    """
    def __init__(self, repo: CalculationRepository):
        self.repo = repo  # Dependency injection

    async def _process_single_job(self, job_id: int, iterations: int) -> JobResult:
        start_time = time.perf_counter()
        
        # Offload the heavy synchronous C computation to a thread pool.
        # Because our C extension releases the GIL, this scales across CPU cores.
        loop = asyncio.get_running_loop()
        computed_value = await loop.run_in_executor(
            None, 
            fast_math.sum_of_squares, 
            iterations
        )
        
        duration = (time.perf_counter() - start_time) * 1000
        return JobResult(
            job_id=f"JOB-{job_id}",
            iterations=iterations,
            calculated_sum=computed_value,
            execution_time_ms=duration
        )

    async def execute_pipeline(self) -> None:
        # 1. Fetch data through interface abstraction
        job_workloads = await self.repo.fetch_pending_jobs()
        
        # 2. Concurrently orchestrate workers via asyncio
        tasks = [
            self._process_single_job(idx, iterations) 
            for idx, iterations in enumerate(job_workloads)
        ]
        
        # Wait for all async jobs to complete
        results: List[JobResult] = await asyncio.gather(*tasks)
        
        # 3. Output results through interface abstraction
        await self.repo.save_results(results)

```

### 5. Clean Architecture: Infrastructure Layer

Save this as `app/infrastructure.py`. This holds concrete implementations (e.g., simulated database engines).

```python
import asyncio
from typing import List
from app.domain import CalculationRepository, JobResult

class InMemoryCalculationRepository:
    """Concrete adapter implementing the CalculationRepository protocol."""
    
    def __init__(self):
        # Simulated heavy DB load values
        self._mock_db = [5_000_000, 8_000_000, 12_000_000, 6_000_000]

    async def fetch_pending_jobs(self) -> List[int]:
        await asyncio.sleep(0.1)  # Simulate network latency I/O
        return self._mock_db

    async def save_results(self, results: List[JobResult]) -> None:
        await asyncio.sleep(0.05)  # Simulate network writing latency I/O
        print(f"\n[Infrastructure] Successfully persisted {len(results)} records to DB:")
        for res in results:
            print(f"  - {res.job_id}: Result={res.calculated_sum} (Took {res.execution_time_ms:.2f}ms)")

```

### 6. Main Runner & Profiler Setup

Save this as `main.py` in your root directory.

```python
import asyncio
import cProfile
import pstats
from app.infrastructure import InMemoryCalculationRepository
from app.use_cases import BatchProcessingUseCase

async def run_application():
    # Instantiate infrastructure dependencies
    repository = InMemoryCalculationRepository()
    
    # Inject dependency into our Core Business logic boundary
    use_case = BatchProcessingUseCase(repo=repository)
    
    print("[Main] Starting Async Clean Architecture Processing Pipeline...")
    await use_case.execute_pipeline()
    print("[Main] Execution pipeline complete.")

def main():
    # Setup cProfile profiling out of the box
    profiler = cProfile.Profile()
    profiler.enable()

    # Bootstrap the async event loop
    asyncio.run(run_application())

    profiler.disable()
    
    # Format and present profiling data
    print("\n" + "="*50 + "\n PROFILING ENGINE RESULTS \n" + "="*50)
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    stats.print_stats(15)  # Print the top 15 most performance-heavy calls

if __name__ == "__main__":
    main()

```

---

## 📖 Comprehensive Project Tutorial

Save this content as `TUTORIAL.md` inside your project directory.

```markdown
# High Performance Python: Architecture, Async & C Extensions

This project serves as an end-to-end blueprint demonstrating how to engineer software using clean structural boundaries, manage concurrent event loops via `asyncio`, write low-level code directly against the CPython API, and profile execution performance.

---

## Architectural Deep Dive

### 1. Clean Architecture & Boundaries
Our core business domain is isolated inside `app/domain.py` and `app/use_cases.py`. 

* **No Framework Contamination:** The business logic knows absolutely nothing about databases, network protocol adapters, or frameworks.
* **Structural Subtyping (`typing.Protocol`):** Instead of concrete classes subclassing rigid Abstract Base Classes, Python uses structural layout (duck typing enforced by static analysis types). If an object contains an async `fetch_pending_jobs` and `save_results` method, it satisfies the boundary. This allows fast mocking without complex setup during test phases.

### 2. Asyncio Orchestration & The GIL
When performing heavy mathematical operations, standard Python execution parks the interpreter on a single core due to the **Global Interpreter Lock (GIL)**.

* In `app/use_cases.py`, we leverage `loop.run_in_executor(None, ...)` to offload compute-heavy work to an internal thread pool.
* Under normal circumstances, multithreading CPU-bound tasks in Python yields zero performance gains. However, because our custom C extension explicitly drops the lock, **true CPU-bound multi-core parallelism is achieved**.

### 3. CPython Extension Mechanics
The extension written inside `c_extensions/fast_math.c` utilizes the native Python C API:
* **`PyArg_ParseTuple`**: Converts Python objects (dynamically typed) safely into native C data types.
* **`Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`**: These macros release and re-acquire the GIL. While the execution remains enclosed inside that specific region, the thread cannot interact with Python objects or call runtime mechanisms, but it can run raw machine code at maximum speed.
* **Reference Counts**: The return call `PyLong_FromLongLong` instantiates a brand new Python integer on the heap and handles reference tracking parameters seamlessly before returning control to the interpreter runtime.

---

## Installation and Execution Guide

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with development tools capable of compiling C (e.g., GCC or Clang on Linux/macOS, or Visual Studio C++ Build Tools on Windows).

### 2. Compile the C Extension
Run the compilation setup from your terminal in the root directory:
```bash
python setup.py build_ext --inplace

```

*This command compiles `fast_math.c` and outputs a `.so` (Linux/macOS) or `.pyd` (Windows) binary directly inside your project folder, allowing standard Python modules to import it via `import fast_math`.*

### 3. Execute the System and Profile

Run the system with profiling enabled:

```bash
python main.py

```

### 4. Interpreting Output

Upon running, you will see output documenting the pipeline state, followed by structural tracing metrics produced via `cProfile`. Look for entries highlighting:

* `method 'sum_of_squares' of 'fast_math' modules`: To see cumulative execution run-time spent down inside your native C block.
* `elapser/selectors`: To view time spent handling non-blocking background network polling loops.

```

---

## 📦 How to create your Zip file
Once you copy the text above into their respective files inside your folder, simply use your operating system's built-in tool to zip it:
* **On Windows:** Right-click the `advanced_python_project` folder ➡️ **Send to** ➡️ **Compressed (zipped) folder**.
* **On macOS/Linux:** Open terminal and run: `zip -r advanced_python_project.zip advanced_python_project/`

```