## Thesis Project: ZEPHYR (Zonal Escape and Potential HYbrid Routing)

This repository contains the implementation and experimental assets for a thesis on autonomous unmanned aerial vehicle (UAV) navigation in cluttered three-dimensional environments. The work investigates a hybrid navigation architecture that combines analytical potential-field methods with reinforcement learning in order to address several limitations of classical artificial potential fields, including oscillatory behavior in constrained passages, weak convergence in locally flat regions, delayed response to dynamic obstacles, and entrapment in local minima.

The proposed framework is organized around three principal components. First, a Gaussian repulsion-based subharmonic field formulation is used to provide smooth and stable goal-directed guidance. Second, an inverse-power subharmonic field formulation with compact support (in the code named superharmonic) is employed to improve responsiveness to time-varying threats through the temporal derivative of the potential. Third, Proximal Policy Optimization (PPO) is used to learn residual or escape behaviors that supplement the analytical controller when field-based navigation alone becomes insufficient. The repository therefore serves as the computational basis for the design, training, evaluation, and analysis of this hybrid system.

## Research Scope

The experimental framework in this repository is intended to support the investigation of four broad research objectives:

- the reduction of oscillatory motion in narrow or geometrically constrained three-dimensional passages,
- improved convergence during long-range navigation and in weak-gradient goal-approach regions,
- earlier and more reliable reaction to fast-moving or approaching obstacles,
- recovery from freeze states, trap configurations, and other failure modes associated with local minima.

To study these questions systematically, the repository includes scenario families, ablation definitions, trained checkpoints, metric-computation utilities, and scripts for statistical post-processing and visualization.

## Repository Structure

### Navigation and control components

The [engines](c:/Users/Teodora/OneDrive/Documents/GitHub/Thesis/engines) directory contains the core implementation of the navigation architecture:

- `subharmonic_field_engine.py` implements the Gaussian subharmonic guidance field.
- `superharmonic_field_engine.py` implements the reactive inverse-power subharmonic field model.
- `hybrid_field_engine.py` contains the switching and blending logic between field regimes.
- `navigation_controller.py` provides shared control and observation-construction utilities.
- `ppo_policy.py` defines the PPO policy model and checkpoint interface.
- `train_ppo_escape.py` contains the training pipeline for the escape-oriented residual policy.
- `integrate_pf_drone.py` connects the navigation stack to the drone simulator for deployment-time experimentation.

### Evaluation framework

The [evaluation](c:/Users/Teodora/OneDrive/Documents/GitHub/Thesis/evaluation) directory contains the experimental framework used to assess the proposed method:

- ablation-study definitions spanning multiple architectural variants,
- three-dimensional scenario families aligned with the thesis research questions,
- per-episode metric extraction and summary aggregation,
- statistical analysis and table-generation utilities,
- scripts for reproducing evaluation outputs and visualization assets.

Generated CSV, JSON, LaTeX, and text files under `evaluation/output` and `evaluation/output/results` document previously executed experiments and their processed results.

### Training assets and auxiliary material

Additional repository contents include:

- trained PPO checkpoints such as `ppo_escape_v10` and `ppo_nav_v9`,
- hyperparameter-search utilities in `tuning`,
- notebooks for environment setup and basic drone experimentation,
- generated figures and scenario renderings in `scenario_visualizations` and `visualizations`.

### Simulation environment

The [gym-pybullet-drones](c:/Users/Teodora/OneDrive/Documents/GitHub/Thesis/gym-pybullet-drones) directory contains the simulator dependency used for both training and deployment-oriented experiments.

## Experimental Workflow

The repository is structured to support a typical research workflow of the following form:

1. install the Python dependencies listed in `requirements.txt` and configure the experimental environment,
2. develop or refine the analytical navigation components in `engines`,
3. train PPO policies for residual escape or end-to-end navigation,
4. execute ablation studies through the evaluation framework,
5. generate aggregate metrics, statistical comparisons, and visual summaries for analysis and reporting.

## Intended Use

This repository is organized as a research codebase rather than as a general-purpose software package. It is therefore primarily intended for:

- reproducing the computational experiments associated with the thesis,
- examining the interaction between analytical field methods and RL-based residual control,
- extending the scenario library or ablation design,
- comparing hybrid navigation against purely analytical or purely learned alternatives,
- generating tables, plots, and supporting material for written reporting.

