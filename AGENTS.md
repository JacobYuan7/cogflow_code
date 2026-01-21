# AGENTS.md - Development Guide for VERL Repository

This file contains essential information for agentic coding agents working in the VERL (Volcano Engine Reinforcement Learning) repository.

## Project Overview

VERL is a sophisticated reinforcement learning training library for large language models with support for distributed training, multiple inference engines (vLLM, SGLang, HuggingFace), and various RL algorithms (PPO, GRPO, SPPO, DAPO, RLOO, ReMax, etc.).

## Essential Commands

### Installation & Setup
```bash
# Install with basic dependencies
pip install -e .

# Install with testing support
pip install -e .[test]

# Install with vLLM support
pip install -e .[test,vllm]

# Install with SGLang support  
pip install -e .[test,sglang]
```

### Code Quality & Linting
```bash
# Run all pre-commit hooks (linting, formatting, checks)
pre-commit run --all-files

# Run specific ruff linting
pre-commit run --all-files ruff

# Run ruff formatting
pre-commit run --all-files ruff-format

# Run type checking (mypy - note: many modules have errors ignored)
pre-commit run --all-files mypy

# Generate and verify trainer configs
scripts/generate_trainer_config.sh
```

### Testing Commands
```bash
# Run all tests
pytest tests/

# Run single test file
pytest tests/path/to/test_file.py

# Run specific test function
pytest tests/path/to/test_file.py::test_function_name

# Run only CPU tests (most logic tests)
pytest tests/ -k "cpu"

# Run distributed tests (requires Ray cluster)
pytest tests/special_distributed/

# Run end-to-end integration tests
pytest tests/special_e2e/

# Check docstring coverage
python3 tests/special_sanity/check_docstrings.py

# Check license headers
python3 tests/special_sanity/check_license.py --directory .
```

## Code Style Guidelines

### File Organization & Structure
- **License Headers**: Every Python file must start with Apache 2.0 copyright header
- **Directory Structure**: Follow modular architecture with clear separation of concerns
- **Configuration Files**: Use hierarchical YAML with Hydra composition patterns
- **Entry Points**: Use `main_*.py` prefix for standalone scripts

### Naming Conventions
```python
# Files and directories: snake_case
example_script.py
config_directory/

# Classes: PascalCase
class ModelWorker:
class BaseRollout:  # Abstract base classes prefixed with "Base"

# Functions and variables: snake_case  
def compute_log_prob():
actor_model_ref

# Constants: UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH
DEFAULT_BATCH_SIZE

# Test files: test_*.py with descriptive names
test_model_worker_on_cpu.py
test_distributed_training_setup.py
```

### Import Organization
```python
# Standard library imports first
import os
import logging
from typing import Optional, Dict, List

# Third-party imports second
import torch
import ray
import hydra
from transformers import AutoModel

# Local imports last
from verl.utils import logger
from verl.models import BaseModel
from verl.workers.actor import ActorWorker
```

### Type Hints
- Encouraged for public APIs but not strictly enforced
- Use Optional for nullable types
- Prefer generic types (Dict, List) over concrete implementations
- Use `# type: ignore` for known mypy issues when necessary

### Code Formatting (Ruff)
- Line length: 120 characters (soft limit)
- Use f-strings for string formatting
- Prefer descriptive variable names over abbreviations
- Use snake_case for all function and variable names
- Class docstrings required for public APIs

### Error Handling Patterns

#### Graceful Degradation
```python
# Handle missing optional dependencies
try:
    import vllm
except ImportError:
    vllm = None
    logging.warning("vLLM not available, falling back to HF inference")

# Validate resources before starting operations
def validate_resources(config):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this operation")
```

#### Distributed Error Handling
```python
# Ray exception propagation
@ray.remote
def distributed_task():
    try:
        return expensive_computation()
    except Exception as e:
        logging.error(f"Task failed: {e}")
        raise  # Re-raise to propagate to caller

# Timeout management
with timeout_context(seconds=1800):
    result = ray.get(remote_task.remote())
```

#### Configuration Validation
```python
# Validate configuration before training
def validate_config(config):
    required_fields = ["model.path", "data.batch_size", "algorithm.learning_rate"]
    for field in required_fields:
        if not hydra.utils.get_method(config, field):
            raise ValueError(f"Missing required config field: {field}")
```

### Testing Patterns

#### Test Structure
```python
# Test file naming: test_<component>_<scenario>_on_cpu.py
# Use descriptive test names
def test_actor_worker_forward_pass_on_cpu():
    """Test that actor worker correctly processes input data."""
    
@pytest.fixture
def sample_config():
    return {"model": {"path": "test/model"}}

def test_with_fixture(sample_config):
    assert sample_config["model"]["path"] == "test/model"
```

#### Testing Categories
- **CPU Tests**: Logic validation, algorithm correctness (`*_on_cpu.py`)
- **GPU Tests**: Distributed functionality, performance validation
- **Integration Tests**: End-to-end workflows
- **Property Tests**: Data structure invariants
- **Mock Tests**: External dependency isolation

#### Best Practices
- Use Ray testing framework for distributed tests
- Mock external services (vLLM, SGLang) when unavailable
- Test both success and failure paths
- Validate configuration parsing and validation

### Configuration Patterns

#### Hydra Configuration Structure
```yaml
# Base configuration with composition
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  - data@data: legacy_data  
  - _self_

# Hierarchical organization
actor_rollout_ref:
  hybrid_engine: true
  timeout: 1800
  
algorithm:
  learning_rate: 1e-6
  clip_ratio: 0.2
```

#### Configuration Management
- Use hierarchical YAML with composition via `defaults`
- Auto-generate reference configs with `scripts/generate_trainer_config.sh`
- Support runtime override via command line
- Validate configurations before use

### Architecture Patterns

#### Plugin System
```python
# Registry-based component discovery
from verl.utils.registry import register

@register(dispatch_mode="dedicated")
class CustomRollout(BaseRollout):
    def rollout(self):
        pass

# Conditional imports based on availability
if is_vllm_available():
    from .vllm_rollout import VLLMRollout
```

#### Distributed Computing
```python
# Ray-based remote execution
@ray.remote(num_gpus=1)
class ActorWorker:
    def __init__(self, config):
        self.config = config
        
    def compute_action(self, states):
        return self.model(states)

# Distributed coordination
ray.init(address="auto")
workers = [ActorWorker.remote(config) for _ in range(num_workers)]
```

### Development Workflow

1. **Setup Environment**: Install with appropriate extras (`[test,vllm]` or `[test,sglang]`)
2. **Run Pre-commit**: `pre-commit run --all-files` before committing
3. **Generate Configs**: `scripts/generate_trainer_config.sh` if modifying trainer configs
4. **Run Tests**: `pytest tests/path/to/test_file.py` for specific functionality
5. **Validate**: Check docstrings and license coverage

### Extension Points

#### Adding New Algorithms
1. Implement in `verl/trainer/ppo/core_algos/` or create new directory
2. Register in configuration system
3. Add tests in `tests/trainer/`
4. Update documentation

#### Adding New Rollout Engines
1. Implement in `verl/workers/rollout/`
2. Inherit from `BaseRollout`
3. Add conditional import in `__init__.py`
4. Add configuration schema

#### Adding New Models
1. Implement in `verl/models/` with model-specific directory
2. Follow existing patterns (llama/, qwen2/, transformers/)
3. Add compatibility tests
4. Update configuration templates

### Performance Considerations

- Use FSDP/FSDP2 for large-scale distributed training
- Leverage vLLM/SGLang for high-throughput inference
- Consider memory-efficient attention implementations
- Use Ray's actor model for distributed coordination
- Profile with `py-spy` for performance bottlenecks

### Debugging Tips

- Check Ray dashboard for cluster status
- Use `logging.setLevel(logging.DEBUG)` for detailed logs
- Verify GPU availability with `torch.cuda.is_available()`
- Test configurations with small models first
- Use `--cfg hydra/job_name=test` for experiment tracking

## Repository-Specific Notes

### Critical Files
- `verl/protocol.py`: Core data transfer protocol - changes affect entire system
- `scripts/generate_trainer_config.sh`: Auto-generates configs - must run after config changes
- `verl/trainer/config/`: Training configurations - modify with care
- `examples/`: Reference implementations - keep updated with API changes

### Dependencies with Versions
- `torch`: Specific versions required for different backends
- `transformers`: HuggingFace integration - version sensitive
- `ray>=2.41.0`: Distributed computing foundation
- `vllm>=0.7.3,<=0.9.1`: High-performance inference with version constraints
- `sglang==0.4.9.post6`: Alternative inference engine with fixed version

### Hardware Support
- NVIDIA GPUs via CUDA
- AMD GPUs via ROCm 
- Ascend NPUs
- CPU-only mode available for development

This guide should help agents navigate the complexity of VERL while maintaining code quality and architectural consistency.