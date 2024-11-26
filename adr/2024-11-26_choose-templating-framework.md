# Jinja2 for templating

## Status

Accepted

## Context

A key part of this project is to generate code for different targets, given a model and parameters. So there's a need for a templating engine for code generation.
There's no particular need for performance, as the generations are one-offs and the bottleneck is unlikely to be the code generation step.
The top considerations are then ease of use and ease of onboarding and maintenance.

We considered Jinja2 and Mako as they are mature, well designed, well documented, with a readable syntax.

Mako has a smaller community around but is actively maintained by sqlalchemy. It allows for more expressive templating, as one can use python straight into the templates.

Jinja2 is more widely in use in the Python community.

## Decision

Use Jinja2. Mako's more expressive templating is more likely to cause maintenance issues than ease any particular pain point.

This having been said, given the project architecture, it's perfectly possible to use different engines for different targets / subtargets.
