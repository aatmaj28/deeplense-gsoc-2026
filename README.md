# Agentic AI for Gravitational Lensing Simulation Workflows

**GSoC 2026 Project** — Machine Learning for Science (ML4SCI) / DeepLense  
**Contributor:** Aatmaj Amol Salunke  
**Mentors:** Michael Toomey (MIT), Sergei Gleyzer (U. Alabama), Pranath Reddy, Rajat Shinde (UAH)

---

## Overview

Current gravitational lensing simulation pipelines require substantial manual intervention: researchers must configure parameters, manage file outputs, validate results, and iterate on failures across multi-step workflows. This bottlenecks large-scale dataset generation and limits exploration of parameter space.

This project builds an **Agentic AI framework** that autonomously orchestrates [DeepLenseSim](https://github.com/ML4SCI/DeepLenseSim) workflows following the **HEPTAPOD philosophy** of schema-validated tools with human-in-the-loop oversight. Researchers can now generate strong gravitational lensing images through natural language interaction — specifying simulation objectives while the agent constructs validated execution plans, manages multi-step pipelines, and produces reproducible structured outputs.

### Key Features

- 🤖 **Natural Language Interface** — Specify simulations in plain English; agent handles the details
- ✅ **Schema-Validated Tools** — Pydantic models enforce parameter constraints *before* simulation execution
- 👤 **Human-in-the-Loop** — Agent asks clarifying questions and requests approval before expensive computations
- 📊 **Structured Outputs** — All results include JSON metadata, image paths, and statistical summaries
- 🔄 **Multi-Step Orchestration** — Parameter scans, comparative studies, and complex workflows
- 🎯 **Supports All DeepLenseSim Configurations** — Model_I, Model_II with no substructure, CDM subhalos, and axion vortices

---

## Architecture

```
User (Natural Language)
    ↓
┌─────────────────────────┐
│  DeepLenseAgent (LLM)   │
│  (OpenRouter/OpenAI)    │
└─────────┬───────────────┘
          ↓
    ┌─────────────────────────────────┐
    │  Schema-Validated Tools         │
    │  • ask_clarification            │ ← Human-in-the-loop
    │  • run_deeplense_simulation     │ ← Pydantic validated
    └────────────┬────────────────────┘
                 ↓
    ┌─────────────────────────────────┐
    │  SimulationRequest (Pydantic)   │
    │  ✓ Constraints validated        │
    │  ✓ Type-checked                 │
    └────────────┬────────────────────┘
                 ↓
    ┌─────────────────────────────────┐
    │  DeepLenseSim (DeepLens class)  │
    │  • Cosmology setup              │
    │  • Halo & substructure config   │
    │  • Simulation execution         │
    └────────────┬────────────────────┘
                 ↓
    ┌─────────────────────────────────┐
    │  SimulationResult (Structured)  │
    │  • Run ID & metadata            │
    │  • Image paths (.npy)           │
    │  • Statistics & JSON output     │
    └─────────────────────────────────┘
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aatmaj28/deeplense-gsoc-2026.git
cd deeplense-gsoc-2026
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv deeplense_env
source deeplense_env/Scripts/activate  # Windows
# or: source deeplense_env/bin/activate  # Linux/Mac

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install openai pydantic numpy matplotlib

# DeepLenseSim and scientific computing
pip install lenstronomy pyHalo

# Jupyter for notebooks
pip install jupyter
```

### 4. Configure OpenRouter API Key

The agent requires an OpenRouter API key to run LLM queries. You have two options:

**Option A: Environment Variable (Recommended)**
```bash
export OPENROUTER_API_KEY='your-api-key-here'
```

**Option B: Update Notebook Directly**
In `Specific_Test_II_Agentic_AI.ipynb`, cell 3, set:
```python
OPENROUTER_API_KEY = 'your-api-key-here'
```

Get your free API key at [openrouter.ai](https://openrouter.ai/).

---

## Quick Start

### Running the Demo

```bash
# Start Jupyter
jupyter notebook

# Open: Specific_Test_II_Agentic_AI.ipynb
# Run all cells to see the agent in action
```

### Using the Agent Programmatically

```python
from notebook_content import DeepLenseAgent

# Create agent instance
agent = DeepLenseAgent()

# Simple request - agent asks for clarification
response = agent.chat("Generate some lensing images for me")
print(response)  # Agent asks: "What substructure type?"

# Provide clarification
response = agent.chat("I want CDM substructure with Model_II, 10 images")
print(response)  # Simulation executes

# Access results
if agent.last_result:
    print(f"Run ID: {agent.last_result.run_id}")
    print(f"Output dir: {agent.last_result.output_dir}")
    print(f"Images generated: {agent.last_result.num_generated}")
```

### Natural Language Examples

```python
# Specific request with all parameters
agent.chat(
    "Generate 8 vortex substructure lensing images using Model_I "
    "with axion mass 1e-23 eV, halo redshift 0.5, source redshift 1.5"
)

# Model_II (Euclid) example
agent.chat(
    "I need 6 smooth lens images (no dark matter substructure) "
    "using the Euclid telescope configuration"
)

# CDM example
agent.chat(
    "Create 5 cold dark matter substructure images with Model_II, "
    "default cosmology, halo at z=0.5, source at z=1.0"
)
```

---

## Project Structure

```
deeplense-gsoc-2026/
├── README.md                                    # This file
├── Specific_Test_II_Agentic_AI.ipynb           # Main evaluation - Agentic AI
├── Common_Test_I_Classification.ipynb           # Evaluation - Classification
├── GSoC_2026_ML4SCI_Proposal_Draft.md          # Full proposal (reference)
├── simulations/                                 # Generated simulation outputs
│   ├── {run_id}/
│   │   ├── metadata.json                       # Simulation parameters & stats
│   │   ├── preview.png                         # Visualization grid
│   │   └── {substructure}_{index}.npy         # Image arrays
│   ├── 1a2e6257/                              # Example: Vortex run
│   └── 387d88c6/                              # Example: No-substructure run
├── DeepLenseSim/                              # Submodule: lensing simulation library
├── pyHalo/                                     # Submodule: halo models
└── deeplense_env/                             # Python virtual environment
```

---

## Key Components

### 1. Pydantic Models (Schema-Validated)

All simulation parameters are validated before execution:

```python
class SimulationRequest(BaseModel):
    substructure_type: SubstructureType        # no_sub, cdm, vortex
    model_config_name: ModelConfig             # Model_I, Model_II
    num_images: int = Field(ge=1, le=100)
    halo_mass: float = Field(default=1e12)
    z_halo: float = Field(default=0.5, gt=0)
    z_source: float = Field(default=1.0, gt=0)
    axion_mass: Optional[float] = None
    cosmology: CosmologyParams
    
    # Automatic validation: source must be behind lens
    @field_validator('z_source')
    def source_behind_lens(cls, v, info):
        if v <= info.data.get('z_halo', 0.5):
            raise ValueError('z_source must be > z_halo')
        return v
```

### 2. Tool Functions

**`run_simulation(request: SimulationRequest) -> SimulationResult`**
- Executes DeepLenseSim with validated parameters
- Generates .npy image arrays
- Returns structured metadata (run_id, statistics, file paths)
- Saves metadata as JSON

**`visualize_results(images, result, max_display=8)`**
- Displays image grid with run metadata
- Saves preview.png

### 3. DeepLenseAgent Class

Core orchestration logic:
- Maintains conversation history
- Calls LLM with function schemas
- Implements `ask_clarification` for human-in-the-loop
- Validates tool outputs before execution
- Manages simulation state

### 4. LLM Configuration

Uses OpenRouter for model-agnostic backend:
- Default model: `google/gemini-2.0-flash-001`
- Easily switch to Claude, GPT-4, Llama, etc.
- Function calling for reliable tool invocation

---

## Supported Simulation Models

| Model | Resolution | PSF | Instrument | Ref |
|-------|-----------|-----|-----------|-----|
| Model_I | 150×150 px | Gaussian | Generic | `simple_sim()` |
| Model_II | 64×64 px | Realistic | Euclid | `simple_sim_2()` |

**Substructure Types:**
- `no_sub` — Smooth lens (CDM below resolution limit)
- `cdm` — Cold dark matter subhalos
- `vortex` — Axion vortex cores (requires `axion_mass` parameter)

---

## Output Specification

### Simulation Result Metadata (JSON)

```json
{
  "run_id": "a1b2c3d4",
  "request": {
    "substructure_type": "vortex",
    "model_config_name": "Model_I",
    "num_images": 8,
    "halo_mass": 1e12,
    "z_halo": 0.5,
    "z_source": 1.5,
    "axion_mass": 1e-23,
    "vortex_mass": 3e10,
    "cosmology": { "H0": 70.0, "Om0": 0.3, "Ob0": 0.05 }
  },
  "num_generated": 8,
  "image_shape": [150, 150],
  "pixel_value_range": [-0.001, 0.085],
  "timestamp": "2026-03-21T14:32:15.123456",
  "output_dir": "simulations/a1b2c3d4",
  "filenames": [
    "vortex_0000.npy", "vortex_0001.npy", ...
  ]
}
```

### File Organization

```
simulations/{run_id}/
├── metadata.json           # Structured simulation metadata
├── preview.png             # Grid visualization of all images
└── {substructure}_{index}.npy  # Individual image arrays
```

---

## Evaluation Results

### Test I: Multi-Class Classification

**Task:** Classify lensing images into substructure types (no_sub, cdm, vortex)  
**Model:** EfficientNet-B0 with transfer learning  
**Results:**
- No Substructure: AUC = 0.979
- Subhalo (CDM): AUC = 0.949
- Vortex: AUC = 0.975
- **Macro-Average AUC: 0.9677** ✅

### Test II: Agentic AI Workflow

**Task:** Build an agentic AI wrapping DeepLenseSim with schema-validated tools  
**Implementation:**
- ✅ Pydantic models for all parameter types
- ✅ Tool functions with full JSON schemas
- ✅ OpenRouter LLM integration
- ✅ Human-in-the-loop confirmation
- ✅ All substructure types (no_sub, cdm, vortex)
- ✅ Both model configurations (Model_I, Model_II)
- ✅ Structured JSON metadata output
- ✅ Interactive demonstrations with visualization

**Results:** Notebook demonstrates end-to-end workflows with natural language interaction, parameter validation, plan confirmation, and reproducible structured outputs. ✅

---

## Development Timeline

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| 1-4 | Core (60h) | SimulationTool, agent core, HITL flow | Working agent with all model configs |
| 5-8 | Advanced (55h) | ParameterScanTool, validation, RL | Multi-point scans, quality checks |
| 9-12 | Wrap-up (60h) | Tests, docs, integration | >80% test coverage, tutorials |

---

## Usage Examples

### Example 1: Single Simulation with Clarification

```python
agent = DeepLenseAgent()

# Initial vague request
response = agent.chat("Generate some lensing images for me")
print(response)
# Output: "Clarification needed: What type of dark matter substructure..."

# Provide details
response = agent.chat("CDM substructure, Model II, 10 images")
# Agent shows plan and executes → generates images, saves metadata
```

### Example 2: Specific Request with All Parameters

```python
agent = DeepLenseAgent()

response = agent.chat(
    "Generate 8 vortex substructure images with axion mass 1e-23 eV, "
    "Model_I, halo mass 1e12 M_sun, z_halo=0.5, z_source=1.5"
)
# Agent validates → shows plan → executes → returns results

visualize_results(agent.last_images, agent.last_result)
```

### Example 3: Parameter Scan (Future)

```python
# Once ParameterScanTool is implemented
response = agent.chat(
    "Run an axion mass sweep: 5 vortex images for each of 10 "
    "log-spaced masses from 1e-24 to 1e-22 eV, Model_I"
)
# Agent orchestrates 50 simulations with progress tracking
```

---

## Dependencies

- **Core:** OpenAI, Pydantic, NumPy, Matplotlib
- **Simulation:** lenstronomy, pyHalo  
- **LLM:** OpenRouter API
- **Optional:** Jupyter (for notebooks)

---

## Citation

If you use this project, please cite:

```bibtex
@software{salunke2026agentic,
  title={Agentic AI for Gravitational Lensing Simulation Workflows},
  author={Salunke, Aatmaj Amol},
  year={2026},
  organization={ML4SCI / DeepLense},
  url={https://github.com/aatmaj28/deeplense-gsoc-2026}
}
```

---

## References

- [HEPTAPOD: Orchestrating HEP Workflows with Agentic AI](https://arxiv.org/abs/2512.15867) — Menzo et al. (2025)
- [DeepLenseSim: Deep Learning for Strong Lensing](https://github.com/ML4SCI/DeepLenseSim)
- [lenstronomy: Gravitational Lens Modeling](https://lenstronomy.readthedocs.io/)
- [pyHalo: Dark Matter Halo Models](https://github.com/dangilman/pyHalo)

---

## Contact

- **Author:** Aatmaj Amol Salunke
- **Email:** aatmajsalunke@yahoo.com
- **GitHub:** [@aatmaj28](https://github.com/aatmaj28)
- **University:** Northeastern University (MS in Artificial Intelligence)

---

## License

This project is part of GSoC 2026 (ML4SCI / DeepLense).  
Details on licensing will be finalized with mentors.
