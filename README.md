# PetriNetBPS - Petri Net Business Process Simulator

A comprehensive Python framework for simulating business processes using Petri Nets.
## Overview

PetriNetBPS is a business process simulator that combines Petri Net modeling with machine learning techniques to generate realistic process simulations. It can discover simulation parameters from historical event logs and simulate new process instances with high fidelity to the original data.

## Features

- **Petri Net Process Simulation**: Execute business processes using Petri Net models
- **Parameter Discovery**: Automatically discover simulation parameters from event logs
- **Machine Learning Integration**: Use ML models for transition probability prediction
- **Resource Management**: Simulate resource availability and scheduling
- **Temporal Modeling**: Support for various time distributions and calendars

## Installation

### Prerequisites

- Python 3.7+
- pip or conda package manager

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda create --name petrinetbps --file requirements.txt
conda activate petrinetbps
```


## Quick Start

### Basic Usage

```python
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
from src.PetriNetBPS import SimulatorParameters, SimulatorEngine

# 1. Load event log and Petri net
log = xes_importer.apply('example_data/purchasing_order.xes')
net, initial_marking, final_marking = pm4py.read_pnml('example_data/purchasing.pnml')

# 2. Create simulation parameters and discover from log
parameters = SimulatorParameters(net, initial_marking, final_marking)
parameters.discover_from_eventlog(log)

# 3. Create and run simulator
simulator = SimulatorEngine(net, initial_marking, final_marking, parameters)
simulated_log = simulator.simulate(n_instances=100)

print(f"Simulated {len(simulated_log['case:concept:name'].unique())} process instances")
```

### Advanced Configuration

```python
# Advanced parameter discovery with ML models
parameters = SimulatorParameters(net, initial_marking, final_marking)
parameters.discover_from_eventlog(
    log,
    mode_ex_time='resource',                    # Execution time modeling mode
    mode_trans_weights='data_attributes',       # Use case attributes for transition weights
    history_weights='count',                    # Include activity history
    model_type='RandomForest',                  # ML model type
    data_attributes=['amount', 'priority'],     # Case attributes to consider
    categorical_attributes=['region']           # Categorical attributes
)

# Run simulation
simulator = SimulatorEngine(net, initial_marking, final_marking, parameters)
simulated_log = simulator.simulate(
    n_instances=500,
    starting_time="2024-01-01 00:00:00",
    remove_head_tail=0.2  # Remove 20% of simulated data for warm-up/cool-down
)
```

## Architecture

### Core Components

#### 1. SimulatorParameters Class

Manages all simulation parameters and discovery methods:

```python
class SimulatorParameters:
    def __init__(self, net, initial_marking, final_marking)
    def discover_from_eventlog(self, log, **kwargs)
```

**Key Parameters:**
- `transition_weights`: Probability weights for transition firing
- `arrival_time_distr`: Distribution for case arrival times
- `arrival_calendar`: Calendar for case arrivals
- `exec_distr`: Execution time distributions for activities
- `roles`: Resource roles and their capabilities
- `role_calendars`: Working calendars for each role

#### 2. SimulatorEngine Class

Executes the simulation process:

```python
class SimulatorEngine:
    def __init__(self, net, initial_marking, final_marking, simulation_parameters)
    def simulate(self, n_instances, **kwargs)
```

### Utility Modules

#### Temporal Utils (`src/temporal_utils.py`)
- Discover execution time distributions
- Find arrival time distributions and calendars
- Handle temporal modeling and scheduling

#### Resource Utils (`src/resources_utils.py`)
- Discover organizational roles from event logs
- Create resource assignments
- Manage resource calendars and availability

#### Distribution Utils (`src/distribution_utils.py`)
- Fit statistical distributions to observed data
- Sample from various distribution types
- Support for: fixed, normal, exponential, uniform, triangular, lognormal, gamma

#### Transition Utils (`src/transitions_utils.py`)
- Build ML models for transition probability prediction
- Compute transition frequencies from event logs
- Handle data attribute integration

#### Control Flow Utils (`src/controlflow_utils.py`)
- Determine enabled transitions in Petri nets
- Manage token flow and marking updates
- Handle transition firing logic

## Configuration Options

### Execution Time Modeling

```python
# Activity-level execution times
mode_ex_time='activity'

# Resource-level execution times
mode_ex_time='resource'
```

### Transition Weight Discovery

```python
# Simple frequency-based weights
mode_trans_weights='frequency'

# ML-based weights using case attributes
mode_trans_weights='data_attributes'
```

### ML Model Types

```python
# Available models for transition prediction
model_type='LogisticRegression'   
model_type='DecisionTreeClassifier' 
model_type='RandomForest'          
```

### History Weight Modes

```python
# No history consideration
history_weights=None

# Count-based history (how many times each activity occurred)
history_weights='count'

# Binary history (whether each activity occurred)
history_weights='binary'
```

## Input Data Format

### Event Log Requirements

The simulator expects event logs in XES format with the following standard attributes:

- `concept:name`: Activity name
- `time:timestamp`: Event timestamp
- `start:timestamp`: Activity start time (optional)
- `org:resource`: Resource identifier
- `org:role`: Resource role
- `case:concept:name`: Case identifier

### Petri Net Format

Petri nets should be in PNML format with:
- Transitions labeled with activity names
- Proper initial and final markings
- Sound net structure

## Output Format

The simulator returns a pandas DataFrame with simulated event log data:

| Column | Description |
|--------|-------------|
| `case:concept:name` | Simulated case ID |
| `concept:name` | Activity name |
| `start:timestamp` | Activity start time |
| `time:timestamp` | Activity end time |
| `org:resource` | Assigned resource |
| `org:role` | Resource role |
| `[other attributes]` | Any additional case attributes |

## Advanced Features

### Custom Resource Calendars

```python
# Define custom working hours
custom_calendar = {
    'ROLE1': {
        'Monday': (9, 17),
        'Tuesday': (9, 17),
        'Wednesday': (9, 17),
        'Thursday': (9, 17),
        'Friday': (9, 17),
        'Saturday': None,  # No work
        'Sunday': None
    }
}
parameters.role_calendars = custom_calendar
```

### Custom Execution Time Distributions

```python
# Define custom distributions for activities
custom_exec_distr = {
    'Activity_A': ('normal', {'loc': 3600, 'scale': 600}),  # mean 1 hour with std 10 min
    'Activity_B': ('exponential', {'loc': 0, 'scale': 1800})  # 30 min average
}
parameters.exec_distr = custom_exec_distr
```

### Resource Availability Management

```python
# Start simulation with specific resource availability
import pandas as pd
from datetime import datetime

resource_availability = pd.Series({
    'Resource_1': datetime(2024, 1, 1, 8, 0),
    'Resource_2': datetime(2024, 1, 1, 9, 0)
})

simulated_log = simulator.simulate(
    n_instances=100,
    resource_availability=resource_availability
)
```

## Examples

See the `PetriNetBPS.ipynb` notebook for complete examples including:

- Basic simulation setup
- Advanced parameter discovery
- Custom configuration scenarios
- Performance analysis
- Result visualization