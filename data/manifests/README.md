# Frozen manifest contract

Actual manifests are not committed until dataset redistribution permission is confirmed.

Download `ntklinhfitus/hwdb-manifest`. Expected files include `master_manifest.csv`, `master_manifest.jsonl`, `train.csv`, `val.csv`, `test.csv`, `dataset_report.csv`, `writer_split.csv`, and `corrupt_images.csv`.

Expected split counts are 6,346 train, 682 validation, and 201 test. Writers must be disjoint.

The locally supplied `hwdb-manifest.zip` inspected during scaffolding has SHA-256:

```text
A57CB39C00EB47BA5D97DE6F40BA823BECB77CB2E160092F7947991084F62F5E
```

The ZIP also contains generated notebook images and should not be committed wholesale.
