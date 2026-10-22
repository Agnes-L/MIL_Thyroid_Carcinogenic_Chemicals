# Thyroid Cancer Exposome: Core Reproducibility Package

This repository contains the two core computational packages used in the thyroid cancer exposome study:

1. `MPL -thyroid target model`: molecular pretraining and thyroid-target fine-tuning resources.
2. `mil-thyroid-cancer-model`: patient-level multiple-instance learning (MIL) input, code, configuration, and final primary results.

All directory and file names added for this repository are English. Only the latest selected versions are included.

## Scope

The package contains source code, model-ready data, selected model weights, model configuration, and final results needed to inspect or reproduce the two core model stages. It deliberately excludes historical trial folders, intermediate checkpoints, applicability-domain analyses, raw instrument files, raw clinical records, and the sample-ID linkage key.

## Privacy

MIL sample identifiers have been replaced by irreversible `Participant_###` identifiers. The original identifier-to-pseudonym key is not included. The case/control label (`active`) is retained because it is required for model reproduction.

## Version control

The MoleSG component uses the final endpoint-selection manifest dated 2026-08-04 and the selected May 2026 endpoint weights. The primary MIL component uses the latest adopted v9 Level 1/2 input and main result run dated 2026-10-09, with the detected-chemical ranking correction.

See the README in each subpackage for an itemized description of its contents.
