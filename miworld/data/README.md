# Production data adapters

Adapters read continuous TextAgent episodes without modifying the public data.
Model-specific spatial and temporal windows are derived here. M3 rendering and
M3 policy use separate adapters because their supervision contracts differ.
