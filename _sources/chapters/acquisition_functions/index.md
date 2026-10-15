# Acquisition Functions for Molecular Optimization

Acquisition functions are critical components of Bayesian optimization that guide the search process by balancing exploration of unknown regions and exploitation of promising areas. This section explores various acquisition functions and their application to molecular optimization.

## What You'll Learn

- The exploration-exploitation tradeoff in molecular optimization
- Different acquisition functions and their mathematical properties
- How to select and implement appropriate acquisition functions for different optimization scenarios

## Chapter Overview

- **Exploration vs. Exploitation**: Understand the fundamental tradeoff between exploring new regions of chemical space and exploiting known promising regions, and how this balance affects optimization efficiency.

- **Expected Improvement**: Learn about the Expected Improvement (EI) acquisition function, one of the most popular strategies that quantifies the expected amount by which a new point will improve upon the current best solution.

- **Upper Confidence Bound**: Explore the Upper Confidence Bound (UCB) acquisition function, which provides a principled way to balance exploration and exploitation through a tunable parameter.

- **Thompson Sampling**: Discover Thompson sampling as an alternative approach to acquisition, which involves drawing random functions from the posterior distribution to guide exploration.

By the end of this section, you'll understand how different acquisition functions work and how to select the most appropriate one for your specific molecular optimization task.