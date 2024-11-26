# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path
import shutil
project_root = Path(__file__).parents[2]
docs_root = project_root / 'docs'
print(project_root)

project = 'BirdNET-Tiny Forge'
copyright = '2024, BirdNET-Team, fold ecosystemics'
author = 'Giovanni Sirio Carmantini, Mehmet Can Işık'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', "sphinx.ext.autosummary", "myst_parser"]
autosummary_generate = True

# heavy deps, slows gen down
autodoc_mock_imports = [
    'keras',
    'keras_tuner',
    'tensorflow',
]
myst_enable_extensions = [
    "html_image",  # Enable HTML image parsing
]
root_img_dir = project_root / 'img'
build_img_dir = docs_root / 'build' / 'img'
build_img_dir.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(root_img_dir, build_img_dir, dirs_exist_ok=True)

html_copy_source = True
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
