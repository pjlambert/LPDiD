#!/bin/bash
# Script to organize the LP-DiD package properly

# Create the package directory structure
mkdir -p lpdid
mkdir -p tests
mkdir -p examples
mkdir -p docs

# Move main modules into the package
mv lpdid.py lpdid/lpdid.py
mv wildboot_fallback.py lpdid/wildboot_fallback.py
mv utils.py lpdid/utils.py
# __init__.py is already created

# Move tests
mv test_lpdid.py tests/

# Move examples
mv example_usage.py examples/
mv example_iv_interactions.py examples/

# Move documentation
mv PACKAGE_STRUCTURE.md docs/
mv ENHANCEMENTS.md docs/
mv FINAL_FEATURES.md docs/

# Create a proper directory structure
echo "Final package structure:"
echo "
lpdid/
├── lpdid/                      # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── lpdid.py              # Main LP-DiD implementation
│   ├── wildboot_fallback.py  # Wild bootstrap fallback
│   └── utils.py              # Utility functions
├── tests/                     # Test directory
│   └── test_lpdid.py         # Unit tests
├── examples/                  # Example scripts
│   ├── example_usage.py      # Basic examples
│   └── example_iv_interactions.py  # Advanced examples
├── docs/                      # Documentation
│   ├── PACKAGE_STRUCTURE.md  # Package structure guide
│   ├── ENHANCEMENTS.md       # Enhancement details
│   └── FINAL_FEATURES.md     # Feature summary
├── setup.py                   # Setup configuration
├── pyproject.toml            # Modern packaging config
├── requirements.txt          # Dependencies
├── README.md                 # Main documentation
├── LICENSE                   # MIT License
├── MANIFEST.in              # Package manifest
├── Makefile                 # Development tasks
└── .gitignore               # Git ignore patterns
"