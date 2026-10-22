# Serum exposomics and multiple instance learning prioritize chemicals associated with thyroid cancer

This repository contains the two core computational packages used in the thyroid cancer exposome study:

1. `MPL -thyroid target model`: molecular pretraining and thyroid-target fine-tuning resources.
2. `mil-thyroid-cancer-model`: patient-level multiple-instance learning (MIL) input, code, configuration, and final primary results.

## Scope

The package contains source code, model-ready data, selected model weights, model configuration, and final results needed to inspect or reproduce the two core model stages. 

## Privacy

MIL sample identifiers have been replaced by irreversible `Participant_###` identifiers.The case/control label (`active`) is retained because it is required for model reproduction.


See the README in each subpackage for an itemized description of its contents.
