# Kedro for pipelining

## Status

Accepted

## Context

The goal of this repo is to go from raw data to trained network to deployable code.
There will be different data sources, potentially hosted in different ways, different network architectures, requiring different preprocessing steps, and so on. The core of the repo will be complex data pipelines.

We have a need to:
- define reusable processing steps and complex pipelines,
- defend against unmaintainable spaghetti pipes by enforcing good software engineering practices

We assume the pipeline will be run on a single machine, with a single GPU. More advanced use cases are outside the scope of this project.

## Decision

Use Kedro as the data pipelining framework of choice.

## Consequences

Bad:
- Playing around with new pipelines becomes perhaps slower, as one needs to take care to follow the Kedro way.
- New contributors unfamiliar with the framework will need to learn at least its basics.
- From experience, Kedro's model is sometimes not rich enough, such that some advanced yet still fairly typical use-cases need workarounds (e.g. data caching between runs, necessary side effects in processing nodes).

Good:
- Code quality will be better.
- More maintainable and reproducible data pipelines.
- Kedro's documentation is top-notch, making on-boarding of new contributors easier.
- Most users should only ever need to modify yaml configurations, rather than code, to get their desired network out.
