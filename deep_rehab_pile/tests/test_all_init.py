"""Unit tests for modules import."""

import ast
import os
import pkgutil

import pytest


@pytest.mark.parametrize(
    "folder_path", ["deep_rehab_pile/classifiers/", "deep_rehab_pile/regressors/"]
)
def test_all_classes_imported(folder_path):
    """Test if all classes imported in init."""
    init_path = os.path.join(folder_path, "__init__.py")

    if not os.path.exists(init_path):
        pytest.fail(f"No __init__.py found in {folder_path}")

    def get_classes_from_module(module_path):
        """Parse a module and return a list of class names defined in it."""
        with open(module_path) as file:
            node = ast.parse(file.read(), filename=module_path)

        classes = [
            n.name
            for n in node.body
            if isinstance(n, ast.ClassDef)
            if n.name != "tAPE"
            and n.name != "Attention_Rel_Scl"
            and n.name != "MySqueezeLayer"
            and n.name != "GCNLayer"
            and n.name != "APE"
        ]
        return classes

    def get_imported_classes_from_init(init_path):
        """Parse the __init__.py file and return a list of imported class names."""
        with open(init_path) as file:
            node = ast.parse(file.read(), filename=init_path)
        imported_classes = []
        for n in node.body:
            if isinstance(n, ast.ImportFrom) and n.module and n.module != "__future__":
                for alias in n.names:
                    imported_classes.append(alias.name.split(".")[0])
        return imported_classes

    imported_classes = get_imported_classes_from_init(init_path)
    all_classes = []

    for _, module_name, is_pkg in pkgutil.iter_modules([folder_path]):
        if not is_pkg:
            module_path = os.path.join(folder_path, f"{module_name}.py")
            all_classes.extend(get_classes_from_module(module_path))

    missing_classes = set(all_classes) - set(imported_classes)
    assert not missing_classes, f"Missing classes in __init__.py: {missing_classes}"
