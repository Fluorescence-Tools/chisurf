---
type: Specification
title: Plugin System — Target
description: The clean-architecture target for extensions — self-describing, manifest-identified plugins reaching the domain only through the facade.
resource: chisurf/plugins/
tags: [target, plugins, manifest, extensibility]
timestamp: '2026-07-05T00:00:00Z'
---

> The clean-architecture target for extensions. Current shape: [plugin system](/architecture/plugin-system.md). Current-state gaps: [assessment](assessment.md).

## Purpose

Almost everything a user does in ChiSurf beyond the bare core is a plugin: the
analysis tools, calculators, viewers, wizards, and importers. This subsystem
defines what a plugin *is*, how the host finds and loads it, and how it is
allowed to touch the rest of the system. It owns *how features are packaged and
integrated*, not what any individual feature computes.

## Design principles

- **Self-describing.** A plugin declares itself in a manifest: identity, what it
  offers, how it is entered, and what it depends on. The host learns everything
  it needs from that declaration without importing the plugin's code.
- **One identity.** A plugin has a single canonical identity from its manifest.
  There are not two competing naming schemes for the same plugin.
- **Reach the domain only through the facade.** Plugins get datasets, fits,
  parameters, and session state through the provided context
  ([RPC & API](rpc.md)) — never by importing core internals or process
  globals. They may use the GUI locally for their own windows.
- **Declarative UI where possible.** A plugin's forms and panels are described as
  data and rendered by the host, so the same declaration works across contexts
  and stays consistent, rather than each plugin hand-building widgets. See the
  [AutoForm framework](/subsystems/gui-autoform.md).
- **Uniform lifecycle.** Discovery, loading, state persistence, and capability
  advertisement work the same for every plugin. Built-in and user plugins follow
  one contract.
- **Honest metadata.** Category, status flags (experimental, deprecated,
  hidden), and declared capabilities reflect reality, because the host surfaces
  them to users and other subsystems.

## Target architecture

```text
manifest (declaration)  ─▶  host discovery  ─▶  registry  ─▶  activation
                                                                  │ receives
                                                          PluginContext ─▶ facade ─▶ domain
```

- **Manifest.** The single declaration of a plugin: its identity, human-readable
  name and category, entry points (GUI / CLI / services), any operations it
  exposes to the RPC layer with their input/output shapes, the events it emits,
  its dependencies, its state-persistence needs, and status flags. Required
  identity fields are always present; the manifest validates against a known
  schema.
- **Discovery & registry.** The host scans the built-in and user plugin
  locations, validates each manifest, and registers the valid ones — reporting,
  not silently swallowing, anything malformed.
- **Activation & context.** When a plugin runs, the host hands it a context that
  exposes the facade and the host services it may use. That context is the
  plugin's entire sanctioned reach into the rest of ChiSurf.
- **UI integration.** A plugin describes its forms and panels declaratively; the
  host renders them. Window and tool state persist through the host's uniform
  mechanism.
- **Capability contribution.** Operations a plugin declares in its manifest
  become part of the RPC surface with documented shapes, so other components and
  automation can call them.

## Rules

1. Every plugin is described by a manifest that validates against the schema and
   carries its required identity fields.
2. A malformed manifest is reported, not silently dropped; a plugin the host
   cannot describe does not load quietly.
3. A plugin has one canonical identity, taken from its manifest.
4. Plugins reach domain and session state only through the provided context /
   facade — never core internals or process globals.
5. Declared metadata — category, entry points, capabilities, status flags —
   MUST match what the plugin actually is and does.
6. Plugin operations exposed to the RPC layer carry documented input/output
   shapes, like any other boundary contract.
7. Built-in and user plugins follow the same discovery, loading, and lifecycle
   contract; demo or experimental plugins are flagged as such.

## Steering notes

Today the plugin layer is a historic mess: two identity conventions coexist (a
legacy module-level name plus the manifest), some manifests are malformed and get
silently dropped at load, one first-class plugin has no manifest at all,
categories disagree with both the directory layout and the display names, and
demo games ship indistinguishably alongside production tools. The target is one
manifest-based identity, validated-and-reported discovery, honest metadata, and a
uniform lifecycle for every plugin. The backlog is the `DATA-01`, `INC-06`, and
`INC-07` findings in [assessment](assessment.md).
