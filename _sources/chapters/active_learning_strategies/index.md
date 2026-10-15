# Active Learning Strategies for Drug Discovery

Active learning strategies determine which experiments to run next to maximize information gain and optimization efficiency. This section explores various strategies for selecting the most informative molecules to test in drug discovery campaigns.

## What You'll Learn

- Different query strategies for selecting informative molecules to test
- Methods for balancing model improvement and finding optimal compounds
- Techniques for efficient exploration of vast chemical spaces

## Chapter Overview

- **Uncertainty Sampling**: Learn how to select molecules where the model is most uncertain, using approaches such as variance-based selection, entropy-based sampling, and confidence intervals.

- **Diversity Sampling**: Explore methods for ensuring diverse exploration of chemical space by selecting molecules that are maximally different from previously tested compounds, including clustering techniques and distance-based approaches.

- **Expected Model Change**: Understand how to select molecules that, if tested, would cause the greatest change to the current model, using techniques such as expected gradient length and query-by-committee.

By the end of this section, you'll have a comprehensive understanding of different active learning strategies and how to apply them to efficiently navigate the vast chemical space in drug discovery projects.