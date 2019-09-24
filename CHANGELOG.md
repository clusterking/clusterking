# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.13.0 - 2019-09-24

### Added

- Configuration of plot labels
- Many improvements and added features for stability tests

## 0.12.0 - 2019-07-23

### Added

- Stability tests

## 0.11.0 - 2019-06-14

### Added

- Scanner: Sample distribution (rather than binning it)
- Configure variable labels on axes

### Changed

- Follow command design pattern and change interfaces accordingly:
  More formal pattern of worker and result classes
- Make physics package its own package
- Make matplotlib dependency optional
- X axis of distribution plots now shows the kinematic variable, rather than the bin number

### Fixed

- Fix calculation of poisson errors
- Compatibility with different matplotlib and pandas versions

## 0.10.0 - 2019-05-16

### Changed

- The two output files ``*_data.csv`` and ``*_metadata.json`` have been replaced by **one** output file in SQL format. This also simplifies loading and saving Data objects and furthermore reduces the required disk space.
