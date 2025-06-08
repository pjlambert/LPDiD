import os

def test_package_structure():
    expected_items = [
        "setup.py",
        "README.md",
        "pyproject.toml",
        "LICENSE.txt",
        "requirements.txt",
        "lpdid",
        "tests",
        "docs",
        "examples",
    ]
    
    workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    actual_items = os.listdir(workspace_path)

    for item in expected_items:
        assert item in actual_items, f"Missing expected file or folder: {item}"

if __name__ == "__main__":
    test_package_structure()
    print("Package structure test passed.")