# TASK.md - MSIConvert Refactoring Roadmap

This document outlines the comprehensive refactoring tasks needed to transform thyra from a functional prototype into a professional, scalable, and collaborative open-source tool for converting mass spectrometry imaging (MSI) datasets to the modern SpatialData/Zarr format.

## ✅ Recent Progress Summary (Completed Tasks)

### Round 1 - Foundation Improvements ✅
- **Enhanced Package Metadata** - Improved PyPI classifiers, URLs, and project information
- **Pre-commit Hooks** - Automated code quality with Black, isort, flake8, bandit
- **Code Coverage CI** - Integrated codecov reporting and coverage thresholds
- **GitHub Templates** - Professional issue and PR templates for better contributions
- **Code of Conduct** - Established community guidelines and expectations
- **Input Validation** - Added robust parameter validation with clear error messages

### Round 2 - Advanced Features ✅
- **Dry-Run Mode** - Added `--dry-run` CLI flag for conversion preview without file output
- **Contributing Guidelines** - Comprehensive CONTRIBUTING.md with development workflow
- **Matrix Testing** - Expanded CI to test Python 3.10, 3.11, and 3.12 across platforms
- **Semantic Versioning** - Documented versioning policy and conventional commit guidelines

### Round 3 - User Experience ✅
- **Professional README** - Comprehensive documentation with examples, badges, and clear structure
- **Unified Progress Bars** - Consolidated multiple confusing progress bars into clear two-phase display
- **Scientific Accuracy** - Progress calculations use non-zero pixels (spectra count) vs total grid pixels
- **Robust Progress Tracking** - Intelligent spectra count detection across different reader types

### Round 4 - Code Quality Automation ✅
- **Pre-commit Hooks Setup** - Automated code formatting with Black, isort, and flake8
- **Git Hook Integration** - Quality checks run automatically before every commit
- **Code Standards Enforcement** - Consistent formatting and linting across the entire codebase
- **Developer Workflow** - Streamlined development process with automated quality assurance

### Round 5 - Metadata and User Experience ✅
- **Explicit Pixel Size Metadata Storage** - Added pixel size metadata to SpatialData.attrs (stored in root .zattrs)
- **Enhanced Interactive Pixel Size Detection** - Fixed incorrect "manual" detection method when user accepts auto-detected values
- **Improved Detection Provenance Tracking** - Now correctly records "automatic_interactive", "manual_override", and "manual" methods
- **Unicode Compatibility Fix** - Resolved Windows encoding issues with Unicode characters in console output
- **SpatialData Standards Compliance** - Metadata now follows SpatialData conventions using object-level attributes

### Round 6 - Code Quality and User Experience ✅
- **Enhanced Input Validation** - Added early CLI validation with file existence checks, format-specific validation (ImzML+.ibd, Bruker .d+analysis files), and descriptive error messages with actionable suggestions
- **Configuration Management Foundation** - Created centralized config.py with all core constants, updated key modules to use centralized configuration
- **Code Quality Improvements** - Further F401 unused import cleanup (54→~15), better module docstrings, cleaner import structure, replaced print() with proper logging in data_processors.py (with required import fix)
- **Professional Error Handling** - Enhanced format detection with helpful suggestions, better CLI parameter validation with specific error context
- **Documentation Fixes** - Fixed FIXME comment in PSI-MS ontology with proper research and accurate obsolete term documentation, recognized comprehensive package metadata already in place

**Note**: Currently using lenient flake8 rules and skipped bandit security scanning due to existing code quality issues. See tasks below for gradual improvements.

**Commit Policy**: All commits should appear as human-authored. No AI co-authorship or AI generation mentions in commit messages.

### Total Completed: 32 major tasks ✅

---

## Task Categories

### Performance & Scalability

- [ ] **Implement Robust Batch Processing Pipeline**
  - **Description:** The current in-memory approach fails for datasets larger than available RAM (100s of GB). This task involves refactoring the core conversion logic to read, process, and write data in managed chunks. The proposed workflow is: 1) Read a batch of spectra from the source reader. 2) Write the raw batch to a temporary on-disk Zarr store with minimal processing. 3) Repeat for all data chunks. 4) Use Dask to perform expensive operations (interpolation, resampling) on the complete on-disk dataset in a memory-efficient manner.
  - **Rationale:** This is the most critical technical improvement. It enables processing of real-world, large-scale MSI datasets that are common in the field, making the tool actually usable for production workloads.
  - **Labels:** `priority:high`, `area:performance`, `area:core-logic`

- [x] **Consolidate Progress Bars into Single User-Friendly Display**
  - **Description:** Currently, users see two separate progress bars (one from the data reader, one from the converter), which creates confusion about actual progress. This task involves analyzing which progress indicator provides more meaningful information to users and consolidating into a single, clear progress bar that shows overall conversion progress with substeps (e.g., "Reading spectra 1000/50000", "Processing batch 5/20", "Finalizing output").
  - **Rationale:** A single, informative progress bar improves user experience and provides clear feedback about the conversion status, especially important for long-running operations on large datasets.
  - **Labels:** `priority:medium`, `area:ux`, `area:performance`, `good-first-issue`

- [ ] **Implement Parallel Processing for Spectrum Operations**
  - **Description:** Many operations on individual spectra (baseline correction, peak picking, etc.) are embarrassingly parallel. This task involves implementing multiprocessing or Dask-based parallelization for spectrum-level operations to utilize all available CPU cores.
  - **Rationale:** Modern workstations have many cores. Parallel processing can significantly reduce conversion time without requiring algorithmic changes.
  - **Labels:** `priority:medium`, `area:performance`

### Data Integrity & Automation

- [ ] **Implement Intelligent Mass Axis Resampling for Bruker Data**
  - **Description:** Bruker datasets often contain mass axes with unrealistically high resolution (e.g., 0.001 Da spacing) that doesn't reflect actual instrument capabilities, leading to unnecessarily large file sizes. This task implements an intelligent resampling algorithm that: 1) **REQUIRES**: Metadata-based bounds detection providing MzAcqRangeLower/Upper from GlobalMetadata and spatial bounds from ImagingArea parameters. 2) Uses mass analyzer physics equations to create optimal non-linear mass axis based on instrument resolution. 3) Implements interpolation during Dask processing or asynchronously during batch reading. 4) Eliminates need to store full high-resolution mass axis in memory by using bounds-only approach. **STRICT PREREQUISITE**: Must complete "Implement Metadata-Based Efficient Bounds Detection" task first - this task cannot function without efficient metadata extraction.
  - **Rationale:** This can reduce output file sizes by 50-90% without losing scientifically relevant information. Memory usage is dramatically reduced by avoiding full mass axis storage. The metadata approach makes this computationally feasible for large datasets.
  - **Labels:** `priority:high`, `area:data-integrity`, `area:performance`, `depends-on:metadata-bounds`

- [ ] **Extract and Preserve Complete Metadata**
  - **Description:** Critical metadata from raw files (instrument model, acquisition parameters, scan settings, etc.) is currently lost during conversion. This task involves: 1) Implementing comprehensive metadata extraction for each supported format. 2) Designing a standardized metadata schema. 3) Embedding all metadata into the SpatialData object's attributes for self-contained, reproducible results.
  - **Rationale:** Metadata is crucial for data reproducibility, quality control, and downstream analysis. A self-contained output file with complete metadata enables better scientific workflows.
  - **Labels:** `priority:high`, `area:data-integrity`

- [x] **Implement Automatic Pixel Size Detection** *(Significantly Enhanced)*
  - **Description:** ✅ **COMPLETED:** Enhanced automatic pixel size detection with proper metadata storage and provenance tracking. Now correctly extracts pixel size from both Bruker and ImzML formats with comprehensive detection method tracking: 1) "automatic_interactive" when user accepts detected size. 2) "manual_override" when user specifies different value after detection. 3) "manual" when no detection possible. 4) Explicit pixel size metadata stored in SpatialData.attrs following standard conventions. 5) Detection info includes source format, detected values, and processing notes. 6) Interactive mode properly preserves detection provenance instead of incorrectly marking as manual input.
  - **Rationale:** Instrument metadata is more accurate than user input. Enhanced implementation now provides complete transparency about detection methods and stores metadata in standardized SpatialData format for better interoperability.
  - **Labels:** `priority:medium`, `area:automation`, `area:ux`, `area:data-integrity`

- [x] **Add Missing Reader Properties for CLI Compatibility**
  - **Description:** ✅ **COMPLETED:** Added @property methods to BaseMSIReader for `shape` (returns dimensions), `n_spectra` (returns total spectrum count), and `mass_range` (returns min/max mass values). Implemented efficiently without triggering expensive full-dataset scans. CLI dry-run functionality now works correctly.
  - **Rationale:** Fixes broken dry-run functionality and provides standard interface for basic dataset information. Essential for CLI usability.
  - **Labels:** `priority:high`, `area:cli`, `area:api-design`, `good-first-issue`

- [ ] **Implement Metadata-Based Efficient Bounds Detection**
  - **Description:** Instead of scanning all spectra/coordinates to determine dataset bounds, leverage metadata tables for immediate extraction. For Bruker data: 1) **Mass bounds**: Extract from GlobalMetadata (MzAcqRangeLower=40.0, MzAcqRangeUpper=600.0) instead of building full mass axis. 2) **Spatial bounds**: Extract from GlobalMetadata (ImagingAreaMinXIndexPos=669, ImagingAreaMaxXIndexPos=837, ImagingAreaMinYIndexPos=109, ImagingAreaMaxYIndexPos=308). 3) **Pixel size analysis**: Extract BeamScanSizeX/Y (46μm) and SpotSize (45μm) from MaldiFrameLaserInfo table, understand efficiency factor vs user input (50μm → 46μm = 92% efficiency). 4) Implement similar metadata extraction for ImzML format. 5) Add fast bounds checking methods to both readers. 6) Create preallocation utilities that use these bounds for efficient memory allocation. **NOTE**: This task is a PREREQUISITE for intelligent mass axis resampling.
  - **Rationale:** This dramatically improves initialization performance by avoiding expensive full-dataset scans. Enables efficient memory preallocation and provides foundation for intelligent interpolation algorithms. Critical for processing large datasets (100+ GB) where scanning all spectra is prohibitively slow. Essential foundation for memory-efficient interpolation methods.
  - **Labels:** `priority:critical`, `area:performance`, `area:optimization`, `area:data-integrity`, `prerequisite-for:interpolation`

### Code Quality & Project Structure

- [x] **Gradually Tighten Flake8 Rules** *(Major Progress - Round 6 Complete)*
  - **Description:** ✅ **FURTHER COMPLETED VIOLATIONS:** F401 (unused imports: 54→39→~15, removed unused os, shapely, spatialdata, xarray, List, BrukerReaderError, numpy imports from key modules). **REMAINING:** F821 (undefined names), E713 (membership tests), F841 (unused variables), E501 (line length violations). Core F401 violations remaining are intentional API exports in __init__.py files.
- [x] **Gradually Tighten Flake8 Rules** *(Major Progress - Round 6 Complete)*
  - **Description:** ✅ **FURTHER COMPLETED VIOLATIONS:** F401 (unused imports: 54→39→~15, removed unused os, shapely, spatialdata, xarray, List, BrukerReaderError, numpy imports from key modules). **REMAINING:** F821 (undefined names), E713 (membership tests), F841 (unused variables), E501 (line length violations). Core F401 violations remaining are intentional API exports in __init__.py files.
  - **Rationale:** Gradual improvement of code quality without disrupting development workflow. Each category should be tackled separately to make progress manageable.
  - **Labels:** `priority:medium`, `area:code-quality`, `area:technical-debt`

- [ ] **Add Bandit Security Scanning to Pre-commit**
  - **Description:** Currently skipped bandit security scanning due to existing issues. This task involves: 1) Running bandit manually to identify all security issues. 2) Fixing legitimate security concerns (XML parsing vulnerabilities, bare except clauses). 3) Adding # nosec comments for false positives with justification. 4) Re-enabling bandit in pre-commit hooks with appropriate skip rules.
  - **Rationale:** Security scanning is important for production code. The issues found (XML vulnerabilities, bare exceptions) should be addressed for robustness.
  - **Labels:** `priority:high`, `area:security`, `area:code-quality`

- [x] **Migrate to Poetry for Dependency Management**
  - **Description:** Modernize the project's dependency management by migrating from requirements.txt to Poetry with pyproject.toml. This includes: 1) Creating a proper pyproject.toml with all dependencies. 2) Setting up development and optional dependency groups. 3) Configuring package metadata and entry points. 4) Updating CI/CD pipelines to use Poetry.
  - **Rationale:** Poetry provides modern, deterministic dependency management with lock files, simplifies virtual environment handling, and makes the project easier to install and contribute to.
  - **Labels:** `priority:high`, `area:code-quality`, `area:infrastructure`

- [ ] **Implement Strict Type Checking with mypy**
  - **Description:** Add comprehensive type hints throughout the codebase and enforce strict type checking using mypy. This includes: 1) Adding type annotations to all functions and classes. 2) Configuring mypy with strict settings. 3) Adding mypy to the CI pipeline. 4) Creating type stubs for any untyped dependencies.
  - **Rationale:** Type checking catches bugs early, improves code maintainability, serves as inline documentation, and makes the codebase more approachable for new contributors.
  - **Labels:** `priority:medium`, `area:code-quality`

- [x] **Create Professional README.md**
  - **Description:** Write a comprehensive README.md that serves as the project's front page. It should include: 1) Clear project description and value proposition. 2) Supported formats and features. 3) Installation instructions (pip, conda, from source). 4) Quick-start usage examples. 5) Links to full documentation. 6) Contributing guidelines summary. 7) Citation information.
  - **Rationale:** The README is the first thing potential users and contributors see. A professional README builds trust and lowers the barrier to adoption.
  - **Labels:** `priority:high`, `area:documentation`, `area:community`, `good-first-issue`

- [ ] **Refactor CLI Module for Better Testability**
  - **Description:** The current thyra_cli.py mixes argument parsing, business logic, and I/O operations. This task involves refactoring to separate concerns: 1) Pure argument parsing layer. 2) Core conversion logic as testable functions. 3) I/O operations isolated in separate functions. 4) Comprehensive unit tests for each layer.
  - **Rationale:** Better separation of concerns improves testability, maintainability, and makes it easier to add new CLI features or alternative interfaces (GUI, API).
  - **Labels:** `priority:medium`, `area:code-quality`, `area:testing`

### Documentation & Community

- [ ] **Set Up Documentation Framework**
  - **Description:** Establish a documentation infrastructure using Sphinx. Start with: 1) Auto-generated API documentation from docstrings. 2) Installation guide. 3) Tutorial for basic conversion workflow. 4) Architecture overview for contributors. Deploy to Read the Docs or GitHub Pages.
  - **Rationale:** Good documentation is essential for user adoption and contributor onboarding. Starting with auto-generated docs provides immediate value while the codebase evolves.
  - **Labels:** `priority:medium`, `area:documentation`, `area:community`

- [x] **Create Contributing Guidelines**
  - **Description:** Write a comprehensive CONTRIBUTING.md that covers: 1) Development environment setup. 2) Code style guidelines. 3) Testing requirements. 4) Pull request process. 5) Issue reporting guidelines. 6) Communication channels.
  - **Rationale:** Clear contribution guidelines lower the barrier for new contributors and ensure consistent code quality across contributions.
  - **Labels:** `priority:medium`, `area:community`, `good-first-issue`

- [x] **Add Code of Conduct**
  - **Description:** Adopt and customize a standard Code of Conduct (e.g., Contributor Covenant) that defines expected behavior and procedures for handling violations.
  - **Rationale:** A Code of Conduct creates a welcoming environment and sets clear expectations for community interactions.
  - **Labels:** `priority:medium`, `area:community`, `good-first-issue`

- [x] **Create Issue and PR Templates**
  - **Description:** Add GitHub issue templates for bug reports, feature requests, and questions. Create a pull request template with a checklist for contributors (tests pass, documentation updated, etc.).
  - **Rationale:** Templates ensure contributors provide necessary information and follow project standards, reducing back-and-forth communication.
  - **Labels:** `priority:low`, `area:community`, `good-first-issue`

### Testing & Validation

- [ ] **Implement Comprehensive Test Suite**
  - **Description:** Expand test coverage to include: 1) Unit tests for all core functions. 2) Integration tests for full conversion workflows. 3) Performance benchmarks. 4) Test data fixtures for all supported formats. 5) Property-based testing for data integrity.
  - **Rationale:** Comprehensive testing prevents regressions, enables confident refactoring, and ensures data conversion accuracy.
  - **Labels:** `priority:high`, `area:testing`, `area:code-quality`

- [ ] **Add Data Validation Framework**
  - **Description:** Implement validation checks to ensure converted data integrity: 1) Mass accuracy verification. 2) Spatial coordinate consistency. 3) Intensity value ranges. 4) Metadata completeness. Include both automated checks and optional validation reports.
  - **Rationale:** Data integrity is paramount in scientific applications. Validation helps users trust the conversion process and catch issues early.
  - **Labels:** `priority:medium`, `area:data-integrity`, `area:testing`

### User Experience Enhancements

- [x] **Add Dry-Run Mode**
  - **Description:** Implement a --dry-run flag that simulates the conversion process without writing output files. It should report: estimated output size, detected format, extracted metadata, and any potential issues.
  - **Rationale:** Allows users to validate their conversion parameters and catch issues before committing to potentially long-running operations.
  - **Labels:** `priority:low`, `area:ux`, `good-first-issue`

- [ ] **Implement Resume Capability for Interrupted Conversions**
  - **Description:** For long-running conversions, implement checkpointing so that interrupted conversions can be resumed from the last successful chunk rather than starting over.
  - **Rationale:** Large dataset conversions can take hours. Resume capability improves reliability and user experience for production use.
  - **Labels:** `priority:medium`, `area:ux`, `area:performance`

### Code Refactoring & Maintainability

- [x] **Extract Configuration Management System** *(Foundation Complete)*
  - **Description:** ✅ **COMPLETED:** Created centralized `thyra/config.py` with all core constants: DEFAULT_BUFFER_SIZE (100000), batch sizes, memory limits (MB_TO_BYTES, LOG_FILE_MAX_SIZE_MB), performance thresholds, pixel size tolerance, and SDK buffer settings. Updated key modules (base_converter.py, __main__.py, logging_config.py) to import from config. **REMAINING:** Full YAML/JSON loading system and validation (phase 2).
- [x] **Extract Configuration Management System** *(Foundation Complete)*
  - **Description:** ✅ **COMPLETED:** Created centralized `thyra/config.py` with all core constants: DEFAULT_BUFFER_SIZE (100000), batch sizes, memory limits (MB_TO_BYTES, LOG_FILE_MAX_SIZE_MB), performance thresholds, pixel size tolerance, and SDK buffer settings. Updated key modules (base_converter.py, __main__.py, logging_config.py) to import from config. **REMAINING:** Full YAML/JSON loading system and validation (phase 2).
  - **Rationale:** Centralized configuration improves maintainability, allows easy tuning for different use cases, and makes the codebase more professional.
  - **Labels:** `priority:high`, `area:code-quality`, `area:maintainability`

- [ ] **Refactor Large Methods into Smaller Units**
  - **Description:** Several methods exceed 100 lines (e.g., `_finalize_data` with 200+ lines, `get_common_mass_axis` with 78 lines). This task involves breaking down complex methods into smaller, focused functions following the Single Responsibility Principle. Important to note that they should be doing one thing and one thing only, and if it takes more lines it's okay as long as we follow the Single Responsibility Principle.
  - **Rationale:** Smaller methods are easier to test, understand, and maintain. They enable better code reuse and make debugging simpler.
  - **Labels:** `priority:medium`, `area:code-quality`, `area:maintainability`

- [ ] **Implement Dependency Injection for Better Testability**
  - **Description:** Current code has hard dependencies on external libraries and file systems. This task involves: 1) Creating interfaces for external dependencies. 2) Implementing dependency injection patterns. 3) Using factory methods for object creation. 4) Enabling easy mocking for unit tests.
  - **Rationale:** Dependency injection improves testability, enables better unit testing, and makes the code more modular and flexible.
  - **Labels:** `priority:medium`, `area:testing`, `area:code-quality`

- [ ] **Extract Coordinate System Abstraction**
  - **Description:** Each reader implements its own coordinate handling logic with code duplication. This task involves: 1) Creating a unified coordinate system interface. 2) Implementing common coordinate transformations. 3) Standardizing coordinate representation across readers.
  - **Rationale:** A unified coordinate system reduces code duplication, prevents coordinate-related bugs, and makes adding new formats easier.
  - **Labels:** `priority:medium`, `area:code-quality`, `area:architecture`

### Error Handling & Robustness

- [ ] **Implement Comprehensive Error Handling Strategy**
  - **Description:** Current error handling is inconsistent (print statements, generic exceptions). This task involves: 1) Creating custom exception hierarchy. 2) Implementing proper error context and chaining. 3) Adding retry mechanisms for transient failures. 4) Providing helpful error messages for users.
  - **Rationale:** Proper error handling improves user experience, makes debugging easier, and increases robustness for production use.
  - **Labels:** `priority:high`, `area:robustness`, `area:ux`

- [x] **Add Input Validation Layer**
  - **Description:** Many functions lack input validation, leading to cryptic errors. This task involves: 1) Adding validation for all public APIs. 2) Checking file formats before processing. 3) Validating parameter ranges and types. 4) Providing clear error messages for invalid inputs.
  - **Rationale:** Input validation prevents crashes, provides better user feedback, and makes the tool more professional.
  - **Labels:** `priority:medium`, `area:robustness`, `area:ux`, `good-first-issue`

- [ ] **Implement Resource Management with Context Managers**
  - **Description:** Some resources (file handles, memory maps) aren't properly managed. This task involves: 1) Implementing context managers for all resources. 2) Ensuring proper cleanup on errors. 3) Adding resource pooling where appropriate.
  - **Rationale:** Proper resource management prevents resource leaks, improves reliability, and enables better performance.
  - **Labels:** `priority:medium`, `area:robustness`, `area:performance`

### API Design & Library Usage

- [ ] **Design Public API for Library Usage**
  - **Description:** Currently focused on CLI usage only. This task involves: 1) Designing a clean public API for programmatic usage. 2) Separating CLI concerns from core functionality. 3) Creating convenience functions for common workflows. 4) Adding API documentation and examples.
  - **Rationale:** A well-designed API enables integration into other tools, increases adoption, and makes the library more versatile.
  - **Labels:** `priority:high`, `area:api-design`, `area:usability`

- [ ] **Implement Streaming API for Large Datasets**
  - **Description:** Add support for streaming processing without loading entire datasets. This involves: 1) Creating iterator-based APIs. 2) Implementing lazy evaluation patterns. 3) Supporting partial dataset processing. 4) Adding streaming examples.
  - **Rationale:** Streaming APIs enable processing of datasets larger than memory and improve performance for selective data access.
  - **Labels:** `priority:medium`, `area:api-design`, `area:performance`

- [ ] **Add Plugin System for Custom Processing Steps**
  - **Description:** Currently, processing steps are hardcoded. This task involves: 1) Creating a plugin interface for custom processors. 2) Implementing a plugin discovery mechanism. 3) Adding hooks for pre/post processing. 4) Documenting the plugin API.
  - **Rationale:** A plugin system enables users to extend functionality without modifying core code, increasing flexibility and adoption.
  - **Labels:** `priority:low`, `area:architecture`, `area:extensibility`

### Cross-Platform & Distribution

- [ ] **Improve Cross-Platform Compatibility**
  - **Description:** Current code has platform-specific paths and dependencies (e.g., Windows DLLs). This task involves: 1) Abstracting platform-specific code. 2) Testing on Windows, Linux 3) Handling path separators correctly. 4) Documenting platform-specific requirements.
  - **Rationale:** Cross-platform support increases user base and makes the tool more professional.
  - **Labels:** `priority:high`, `area:compatibility`, `area:distribution`

- [ ] **Create Conda Package**
  - **Description:** Add conda-forge packaging for easier installation, especially for scientific users. This involves: 1) Creating conda recipe. 2) Setting up conda-forge feedstock. 3) Handling binary dependencies. 4) Testing conda installation.
  - **Rationale:** Many scientific users prefer conda for managing complex dependencies. Conda packaging improves accessibility.
  - **Labels:** `priority:medium`, `area:distribution`, `area:community`

- [ ] **Add Docker Support**
  - **Description:** Create Docker images for consistent environments. This involves: 1) Creating Dockerfile with all dependencies. 2) Optimizing image size. 3) Supporting different base images. 4) Publishing to Docker Hub.
  - **Rationale:** Docker support enables consistent environments, simplifies deployment, and helps with reproducibility.
  - **Labels:** `priority:low`, `area:distribution`, `area:devops`

### Security & Privacy

- [x] **Remove Usage of python-dotenv in Production**
  - **Description:** ✅ **COMPLETED:** Removed the dotenv import and load_dotenv() call from __main__.py. The code no longer auto-loads .env files in production, improving security.
  - **Rationale:** Production code shouldn't auto-load .env files for security reasons. Proper configuration management is more secure.
  - **Labels:** `priority:medium`, `area:security`, `area:code-quality`

- [ ] **Add Security Scanning to CI Pipeline**
  - **Description:** Implement automated security scanning. This involves: 1) Adding dependency vulnerability scanning. 2) Implementing code security analysis. 3) Setting up security alerts. 4) Creating security policy.
  - **Rationale:** Proactive security scanning prevents vulnerabilities and builds user trust.
  - **Labels:** `priority:medium`, `area:security`, `area:infrastructure`

### Performance Optimization

- [ ] **Implement Smart Caching Strategy**
  - **Description:** Add caching for expensive operations. This involves: 1) Identifying cacheable computations. 2) Implementing memory-efficient caching. 3) Adding cache invalidation logic. 4) Making cache configurable.
  - **Rationale:** Caching can significantly improve performance for repeated operations and large datasets.
  - **Labels:** `priority:medium`, `area:performance`, `area:optimization`

- [ ] **Optimize Memory Usage with Generators**
  - **Description:** Replace list comprehensions with generators where appropriate. This involves: 1) Identifying memory-intensive operations. 2) Converting to generator-based processing. 3) Adding memory profiling tests. 4) Documenting memory usage patterns.
  - **Rationale:** Generators reduce memory footprint and enable processing of larger datasets.
  - **Labels:** `priority:medium`, `area:performance`, `area:optimization`

### Developer Experience

- [x] **Add Pre-commit Hooks**
  - **Description:** Implement pre-commit hooks for code quality. This involves: 1) Setting up pre-commit framework. 2) Adding hooks for black, isort, flake8, mypy. 3) Including security checks. 4) Documenting setup process.
  - **Rationale:** Pre-commit hooks ensure code quality before commits, reducing review cycles and maintaining consistency.
  - **Labels:** `priority:low`, `area:developer-experience`, `area:code-quality`, `good-first-issue`

- [ ] **Create Development Container Configuration**
  - **Description:** Add devcontainer.json for VS Code and GitHub Codespaces. This involves: 1) Creating container configuration. 2) Installing all development dependencies. 3) Setting up extensions and tools. 4) Testing in multiple environments.
  - **Rationale:** Development containers provide consistent environments and lower the barrier for new contributors.
  - **Labels:** `priority:low`, `area:developer-experience`, `area:infrastructure`

- [ ] **Implement Comprehensive Logging Strategy**
  - **Description:** Current logging is minimal and inconsistent. This involves: 1) Adding structured logging throughout. 2) Implementing log levels appropriately. 3) Adding performance logging. 4) Creating log analysis tools.
  - **Rationale:** Good logging is essential for debugging, monitoring, and understanding system behavior in production.
  - **Labels:** `priority:medium`, `area:observability`, `area:maintainability`

### CI/CD & Release Management

- [x] **Add Code Coverage to CI Pipeline**
  - **Description:** Current tests don't report coverage. This involves: 1) Adding coverage reporting to pytest runs. 2) Integrating with coverage services (Codecov/Coveralls). 3) Setting coverage thresholds. 4) Adding coverage badges to README.
  - **Rationale:** Coverage metrics help identify untested code and maintain quality standards.
  - **Labels:** `priority:medium`, `area:testing`, `area:infrastructure`, `good-first-issue`

- [x] **Implement Matrix Testing for Python Versions**
  - **Description:** Currently only testing on Python 3.12. This involves: 1) Expanding test matrix to include Python 3.10, 3.11, 3.12. 2) Testing on multiple OS versions. 3) Adding compatibility checks. 4) Documenting supported versions.
  - **Rationale:** Broader Python version support increases user base and catches version-specific issues early.
  - **Labels:** `priority:medium`, `area:testing`, `area:compatibility`

- [ ] **Add Integration Tests to CI**
  - **Description:** CI currently skips integration tests. This involves: 1) Creating separate CI job for integration tests. 2) Setting up test data fixtures. 3) Running on schedule or manual trigger. 4) Caching test data for efficiency.
  - **Rationale:** Integration tests catch real-world issues that unit tests miss, ensuring end-to-end functionality.
  - **Labels:** `priority:high`, `area:testing`, `area:infrastructure`

- [ ] **Improve Release Workflow**
  - **Description:** Current release workflow has outdated Python version and manual steps. This involves: 1) Updating to latest Python and action versions. 2) Adding automated PyPI publishing. 3) Creating release notes from CHANGELOG. 4) Adding release validation steps.
  - **Rationale:** Automated releases reduce errors and ensure consistent distribution process.
  - **Labels:** `priority:medium`, `area:infrastructure`, `area:release-management`

### Package Quality & Standards

- [x] **Add Package Metadata and Classifiers**
  - **Description:** ✅ **COMPLETED:** Comprehensive metadata already present in pyproject.toml including: 1) Complete author and maintainer information. 2) Project URLs (homepage, repository, documentation, bug tracker, discussions, changelog, contributing). 3) Detailed PyPI classifiers covering development status, audience, topics, and Python versions. 4) Comprehensive keywords for discoverability (12 relevant terms covering mass spectrometry, spatial omics, data conversion).
  - **Rationale:** Complete metadata improves package discoverability and provides users with essential information.
  - **Labels:** `priority:low`, `area:packaging`, `area:documentation`, `good-first-issue`

- [x] **Implement Semantic Versioning Properly**
  - **Description:** Version management needs improvement. This involves: 1) Documenting version policy. 2) Adding version bumping scripts. 3) Ensuring API stability between versions. 4) Creating migration guides for breaking changes.
  - **Rationale:** Proper versioning helps users understand compatibility and plan upgrades.
  - **Labels:** `priority:medium`, `area:release-management`, `area:documentation`

- [ ] **Add Benchmarking Suite**
  - **Description:** No performance benchmarks exist. This involves: 1) Creating benchmark suite for key operations. 2) Adding performance regression tests. 3) Tracking performance over versions. 4) Publishing benchmark results.
  - **Rationale:** Benchmarks ensure performance improvements and catch regressions early.
  - **Labels:** `priority:low`, `area:performance`, `area:testing`

### Architecture & Design Patterns

- [ ] **Implement Factory Pattern for Readers/Converters**
  - **Description:** Current registry uses global dictionaries. This involves: 1) Creating factory classes. 2) Implementing proper dependency injection. 3) Supporting dynamic loading. 4) Improving error messages.
  - **Rationale:** Factory pattern provides better encapsulation and makes testing easier.
  - **Labels:** `priority:medium`, `area:architecture`, `area:code-quality`

- [ ] **Add Observer Pattern for Progress Updates**
  - **Description:** Progress reporting is tightly coupled. This involves: 1) Creating progress observer interface. 2) Implementing event-based updates. 3) Supporting multiple observers. 4) Enabling custom progress handlers.
  - **Rationale:** Observer pattern decouples progress reporting and enables flexible UI integration.
  - **Labels:** `priority:low`, `area:architecture`, `area:ux`

- [ ] **Create Abstract Base Classes for Common Operations**
  - **Description:** Repeated patterns lack abstraction. This involves: 1) Identifying common operations. 2) Creating abstract base classes. 3) Implementing template methods. 4) Reducing code duplication.
  - **Rationale:** Proper abstractions reduce duplication and enforce consistent interfaces.
  - **Labels:** `priority:medium`, `area:architecture`, `area:maintainability`

## Implementation Priority

1. **Critical (Do First):**
   - Implement Metadata-Based Efficient Bounds Detection (prerequisite for interpolation)
   - Implement Robust Batch Processing Pipeline
   - Extract Configuration Management System
   - Improve Cross-Platform Compatibility

2. **High Priority (Quick Wins):**
   - ✅ Add Missing Reader Properties for CLI Compatibility (fixes dry-run crashes) - COMPLETED
   - ✅ Implement Automatic Pixel Size Detection (leverages new metadata insights) - COMPLETED

3. **High Priority (Major Features):**
   - Implement Intelligent Mass Axis Resampling (REQUIRES bounds detection first)
   - Extract and Preserve Complete Metadata
   - Implement Comprehensive Test Suite
   - Design Public API for Library Usage
   - Implement Comprehensive Error Handling Strategy

4. **Medium Priority:**
   - All other refactoring and optimization tasks
   - Documentation and community tasks
   - Performance optimizations

4. **Low Priority:**
   - Developer experience enhancements
   - Advanced features like plugin systems
   - Distribution improvements (Docker, etc.)

## Notes for Contributors

- Tasks marked with `good-first-issue` are suitable for newcomers to the project
- Performance-related tasks should include benchmarks before and after implementation
- All changes should maintain backward compatibility where possible
- Consider creating feature branches for major refactoring work
- Follow the existing code style and add tests for all new functionality
- Update documentation when adding new features or changing APIs
