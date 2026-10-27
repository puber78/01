"""proxy_selection: research code for digital-proxy detector selection.

Scope (EXAM01 freeze): offline selection of ONE frozen single-image RGB
detector, original photographic (label 0) vs AI-generated (label 1) source
labels, utility = real-recapture AUC on source groups.

Explicitly out of scope for this codebase: active capture policies, per-image
routing/fusion, and training new detector backbones.
"""

__version__ = "0.1.0"
