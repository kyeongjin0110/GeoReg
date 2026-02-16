# GeoReg: Weight-Constrained Few-Shot Regression for Socio-Economic Estimation using LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GeoReg is an interpretable and scalable regression framework designed to estimate socio-economic indicators (e.g., Regional GDP, Population, Education levels) in data-scarce regions. By leveraging Large Language Models (LLMs) as "data engineers," GeoReg extracts informative features from heterogeneous data sources (satellite imagery and web-based geospatial info) and applies domain-informed weight constraints to prevent overfitting in few-shot settings.

## Key Features

* Data-Efficient (Few-Shot): Achieves robust performance even with highly limited ground-truth labels (e.g., 3-shot or 5-shot settings).
* LLM-Driven Inductive Bias: Uses LLMs to categorize feature-target correlations (Positive, Negative, Mixed) and discover complex non-linear feature interactions without needing massive training data.
* Highly Interpretable: Employs a linear regression backbone with strict weight constraints. You can transparently see exactly how and why specific features (like nightlight or distance to airports) affect the predictions.
* Multi-Modal Integration: Seamlessly combines satellite imagery (OpenEarthMap, VIIRS) with geospatial attributes (ArcGIS, Natural Earth).

## Model Architecture

The GeoReg pipeline operates in two main stages:

### Stage 1: Knowledge-based Module Categorization & Feature Discovery
Before training, GeoReg defines various modules to extract structured information (e.g., get_area, get_night_light, count_area). 
The LLM evaluates the correlation between each module and the target indicator, categorizing them into:
* Positive (P)
* Negative (N)
* Mixed (M)
* Irrelevant (Discarded)

The LLM also identifies meaningful non-linear feature interactions to capture hidden socio-economic dynamics.

### Stage 2: Linear Regression with Weight Constraints
A linear regression model is trained using the selected features. To prevent overfitting and align the model with real-world economic logic, weights (beta) are strictly constrained based on the LLM's categorization:
* beta > 0 for Positive features
* beta < 0 for Negative features
* Unconstrained for Mixed features

## Getting Started

### Prerequisites
* Python 3.8+
* OpenAI API Key (for GPT-3.5-turbo / feature discovery)

### Data Preparation
Place your pre-processed geospatial and satellite data in the data/ directory. The expected format is:
* data/features/: Contains extracted module outputs (CSV/JSON).
* data/labels/: Contains few-shot ground-truth indicators (GRDP, Population, etc.).

## Usage

1. Run LLM Module Categorization & Feature Discovery
This step requires an active OpenAI API key.
export OPENAI_API_KEY="your-api-key-here"
python run_stage1_llm.py --country KOR --indicator POP

2. Train and Evaluate the Regression Model
Train the weight-constrained linear regression model using the outputs from Stage 1.
python run_stage2_regression.py --shots 5 --ensemble_size 5

## Results

GeoReg demonstrates superior performance (average winning rate of 87.2%) compared to traditional regression, visual representation models, and direct LLM inferences across diverse countries (South Korea, Vietnam, Cambodia).
