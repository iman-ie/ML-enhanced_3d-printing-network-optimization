# 3D Printing Recycling Network Optimization

A machine learning-enhanced optimization system for designing efficient 3D printing recycling networks. This project combines neural network regression with mixed-integer programming to optimize facility location decisions across multiple facility types.

## 🎯 Project Overview

This system optimizes the placement of five types of facilities in a 3D printing recycling network:
- **3D Printing Centers (DPCs)** - Manufacturing facilities
- **Treatment Centers (TCs)** - Material processing facilities  
- **Recycling Centers (RCs)** - Waste recycling facilities
- **Processing Centers (PCs)** - Material preparation facilities
- **Collection Centers (CCs)** - Material collection points

The optimization combines:
- **Machine Learning**: Neural network regression for cost prediction
- **Mathematical Optimization**: Mixed-integer programming with Gurobi
- **Comprehensive Analysis**: Performance metrics and visualizations

## 🚀 Features

- **Hybrid ML-Optimization Approach**: Integrates neural networks with mathematical optimization
- **Multi-Facility Optimization**: Simultaneously optimizes 5 different facility types
- **Performance Analytics**: Comprehensive ML model evaluation metrics
- **Rich Visualizations**: Learning curves, cost breakdowns, facility utilization analysis
- **Modular Architecture**: Clean, object-oriented design for easy extension
- **Results Export**: JSON export functionality for further analysis

## 📋 Requirements

### Dependencies
```
gurobipy>=10.0.0
gurobi-ml>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
```

### System Requirements
- **Gurobi License**: Academic or commercial license required
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: At least 100MB free space

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/3d-printing-network-optimization.git
   cd 3d-printing-network-optimization
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Gurobi license**
   - Obtain a Gurobi license from [Gurobi website](https://www.gurobi.com/)
   - Follow Gurobi installation instructions for your system

## 📊 Data Format

### Input Files Required
- **X_data20.csv**: Feature matrix (training inputs)
- **y_data20.csv**: Target values (training outputs)

### Data Structure
- Features should be numerical values representing network parameters
- Targets should be cost values or performance metrics
- CSV files should have no headers (header=None in pandas)

## 🚦 Usage

### Basic Usage
```python
from network_optimizer import NetworkOptimizer

# Initialize optimizer
optimizer = NetworkOptimizer()

# Load your data
X_data, y_data = optimizer.load_data("X_data20.csv", "y_data20.csv")

# Train ML model
optimizer.train_ml_model(X_data, y_data)

# Build and solve optimization model
optimizer.build_optimization_model()
results = optimizer.solve_optimization()

# Create visualizations
optimizer.create_visualizations()

# Save results
optimizer.save_results("output_directory")
```

### Command Line Usage
```bash
# Run the complete optimization pipeline
python network_optimizer.py

# Make sure your data files are in the same directory or update paths in main()
```

### Advanced Configuration
```python
# Custom ML configuration
config = {
    'hidden_layers': [20, 20, 20],  # Custom neural network architecture
    'max_iter': 1000,               # Training iterations
    'alpha': 0.001,                 # Regularization parameter
    'train_size': 0.8              # Training data proportion
}

optimizer = NetworkOptimizer(config)
```


### Class Structure
```
NetworkOptimizer/
├── __init__()              # Initialize parameters and configuration
├── load_data()             # Data loading and preprocessing
├── train_ml_model()        # Neural network training
├── build_optimization_model() # Gurobi model construction
├── solve_optimization()    # Model solving and result extraction
├── create_visualizations() # Comprehensive plotting
└── save_results()          # Export results to files
```

### Key Components
- **ML Pipeline**: Polynomial features + MLP regression
- **Optimization Model**: Mixed-integer programming with ML constraints
- **Visualization Engine**: Matplotlib/Seaborn-based plotting system
- **Results Management**: JSON export and analysis tools

## 🔧 Configuration

### Facility Network Parameters
```python
# Default network structure (customizable)
I = 4   # Number of 3D Printing Centers
J = 4   # Number of Treatment Centers  
K = 8   # Number of Recycling Centers
T = 6   # Number of Processing Centers
L = 4   # Number of Collection Centers
```

### ML Model Parameters
```python
# Neural network configuration
hidden_layers = [18] * 7    # 7 layers with 18 neurons each
activation = 'relu'         # Activation function
max_iter = 850             # Maximum training iterations
alpha = 0.0012             # L2 regularization parameter
```
