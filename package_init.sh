#!/bin/bash
# Script to ensure the LP-DiD package is properly structured and ready for development

echo "========================================="
echo "LP-DiD Package Setup Script"
echo "========================================="

# Check if we're in the right directory
if [ ! -d "LPDiD" ] || [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the root of the LP-DiD repository"
    exit 1
fi

# Create necessary directories if they don't exist
echo "Checking directory structure..."
mkdir -p examples
mkdir -p docs

# Clean up any __pycache__ directories
echo "Cleaning up cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Check for duplicate LICENSE files
if [ -f "LICENSE" ] && [ -f "LICENSE.txt" ]; then
    echo "Note: Both LICENSE and LICENSE.txt exist. Consider removing one."
fi

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Optionally install development dependencies
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[dev]"
fi

# Optionally install wild bootstrap dependency
read -p "Install wild bootstrap dependency? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[wildbootstrap]"
fi

# Display current package structure
echo
echo "Current package structure:"
echo "========================="
tree -L 2 -I '__pycache__|*.egg-info|.git|.pytest_cache' 2>/dev/null || {
    # Fallback if tree is not installed
    echo "LPDiD/"
    echo "├── LPDiD/                      # Main package directory"
    echo "│   ├── __init__.py            # Package initialization"
    echo "│   ├── lpdid.py              # Main LP-DiD implementation"
    echo "│   ├── spawn_enforcer.py     # Multiprocessing spawn enforcement"
    echo "│   ├── wildboot_fallback.py  # Wild bootstrap fallback"
    echo "│   └── utils.py              # Utility functions"
    echo "├── internal_tests/            # Test directory"
    echo "│   ├── test_basic_functionality.py"
    echo "│   └── apply_lpdid_to_synth_data.ipynb"
    echo "├── docs/                      # Documentation"
    echo "├── examples/                  # Example scripts"
    echo "├── references/                # Reference papers"
    echo "├── pyproject.toml            # Modern packaging config"
    echo "├── setup.py                  # Setup configuration"
    echo "├── requirements.txt          # Dependencies"
    echo "├── README.md                 # Main documentation"
    echo "└── .gitignore               # Git ignore patterns"
}

echo
echo "Setup complete!"
echo
echo "To run tests:"
echo "  python -m pytest internal_tests/test_basic_functionality.py"
echo
echo "To use the package:"
echo "  from LPDiD import LPDiD"
echo
echo "See README.md and QUICKSTART.md for usage examples."
