# MSIConverter Technical Debt Remediation - Task List

## 1. Logging System Implementation
### Priority: HIGH
- [x] **Create centralized logging configuration module** (`msiconvert/utils/logging_config.py`)
  - [x] Configure file-based logging with rotation
  - [x] Set up different log levels for console and file output
  - [x] Create separate loggers for different modules
  - [x] Add structured logging format with contextual information
- [x] **Replace print statements with proper logging**
  - [x] Update `__main__.py` to use logger instead of print statements
  - [x] Add logging to all reader and converter classes
  - [x] Ensure error tracebacks are properly logged
- [x] **Add logging configuration to CLI**
  - [x] Add `--log-file` parameter to specify log file location
  - [x] Implement log rotation based on size/date

## 2. Code Organization and Responsibility Separation
### Priority: HIGH

### 2.1 Refactor Large Files
- [ ] **Split `spatialdata_converter.py`** (currently handling too many responsibilities)
  - [ ] Extract shape creation logic to `msiconvert/converters/utils/shape_builder.py`
  - [ ] Extract metadata handling to `msiconvert/converters/utils/metadata_handler.py`
  - [ ] Extract image processing to `msiconvert/converters/utils/image_processor.py`
  - [ ] Keep main converter focused on orchestration only

- [ ] **Split `bruker_reader.py`** (mixing DLL handling with data reading)
  - [ ] Extract DLL management to `msiconvert/readers/bruker/dll_manager.py`
  - [ ] Extract SQL queries to `msiconvert/readers/bruker/sql_queries.py`
  - [ ] Create separate classes for TSF and TDF handling

### 2.2 Create Clear Module Structure
- [ ] **Reorganize converter utilities**
  ```
  msiconvert/converters/
  ├── __init__.py
  ├── spatialdata_converter.py
  └── utils/
      ├── __init__.py
      ├── shape_builder.py
      ├── metadata_handler.py
      ├── image_processor.py
      └── data_finalizer.py
  ```

- [ ] **Reorganize reader utilities**
  ```
  msiconvert/readers/
  ├── __init__.py
  ├── imzml_reader.py
  ├── bruker_reader.py
  └── bruker/
      ├── __init__.py
      ├── dll_manager.py
      ├── sql_queries.py
      ├── tsf_handler.py
      └── tdf_handler.py
  ```

## 3. Error Handling and Recovery
### Priority: MEDIUM
- [ ] **Create custom exception hierarchy**
  - [ ] `MSIConvertError` (base exception)
  - [ ] `ReaderError`, `ConverterError`, `ValidationError`
  - [ ] `DLLNotFoundError`, `DataCorruptionError`
- [ ] **Implement graceful degradation**
  - [ ] Add recovery mechanisms for partial data corruption
  - [ ] Implement checkpointing for large conversions
  - [ ] Add ability to resume failed conversions
- [ ] **Improve error messages**
  - [ ] Add contextual information to all exceptions
  - [ ] Provide actionable suggestions for common errors

## 4. Configuration Management
### Priority: MEDIUM
- [ ] **Create configuration system**
  - [ ] Add `msiconvert/config.py` for default settings
  - [ ] Support configuration files (YAML/TOML)
  - [ ] Allow environment variable overrides
  - [ ] Create configuration validation
- [ ] **Externalize hardcoded values**
  - [ ] Buffer sizes (currently hardcoded as 100000)
  - [ ] Chunk sizes for processing
  - [ ] Default compression levels
  - [ ] Timeout values for DLL operations

## 5. Performance Optimization
### Priority: MEDIUM
- [ ] **Implement progress tracking**
  - [ ] Add proper progress callbacks to base classes
  - [ ] Implement ETA calculation
  - [ ] Add memory usage monitoring
- [ ] **Optimize memory usage**
  - [ ] Implement streaming for large datasets
  - [ ] Add option for out-of-core processing
  - [ ] Profile and optimize memory hotspots
- [ ] **Add parallel processing support**
  - [ ] Implement concurrent spectrum processing
  - [ ] Add multiprocessing for independent operations
  - [ ] Ensure thread safety in shared resources

## 6. Testing Improvements
### Priority: HIGH
- [ ] **Increase test coverage**
  - [ ] Add integration tests for error scenarios
  - [ ] Add performance benchmarks
  - [ ] Add tests for edge cases (empty files, corrupted data)
- [ ] **Improve test organization**
  - [ ] Add fixtures for common test scenarios
  - [ ] Create test data generators
  - [ ] Add parameterized tests for multiple formats
- [ ] **Add test documentation**
  - [ ] Document test setup requirements
  - [ ] Add examples for running specific test suites

## 7. Documentation and Code Quality
### Priority: LOW
- [ ] **Improve code documentation**
  - [ ] Add comprehensive docstrings to all public methods
  - [ ] Create architecture documentation
  - [ ] Add code examples in docstrings
- [ ] **Add type hints**
  - [ ] Complete type annotations for all functions
  - [ ] Add mypy to CI pipeline
  - [ ] Fix existing type inconsistencies
- [ ] **Code quality tools**
  - [ ] Set up pre-commit hooks
  - [ ] Configure pylint/flake8 rules
  - [ ] Add code complexity checks

## 8. API Design Improvements
### Priority: LOW
- [ ] **Create cleaner public API**
  - [ ] Define clear public vs private interfaces
  - [ ] Add API versioning support
  - [ ] Create facade classes for complex operations
- [ ] **Add plugin system**
  - [ ] Allow custom readers/converters as plugins
  - [ ] Create plugin discovery mechanism
  - [ ] Add plugin validation

## 9. Specific Technical Debt Items
### Priority: MEDIUM
- [ ] **Fix warning suppressions**
  - [ ] Properly handle `pyimzml.ontology` warnings instead of suppressing
  - [ ] Address `CryptographyDeprecationWarning`
- [ ] **Remove code duplication**
  - [ ] Consolidate common patterns in readers
  - [ ] Extract shared validation logic
  - [ ] Unify error handling patterns
- [ ] **Update deprecated dependencies**
  - [ ] Review and update all dependencies
  - [ ] Replace deprecated API usage
  - [ ] Add dependency security scanning

## 10. Monitoring and Observability
### Priority: LOW
- [ ] **Add metrics collection**
  - [ ] Conversion success/failure rates
  - [ ] Performance metrics (time, memory)
  - [ ] File format statistics
- [ ] **Create health checks**
  - [ ] DLL availability check
  - [ ] Dependency version checks
  - [ ] Disk space validation

## Implementation Order Recommendation

1. **Phase 1 (Immediate)**: Logging System, Error Handling
2. **Phase 2 (Short-term)**: Code Organization, Test Coverage
3. **Phase 3 (Medium-term)**: Configuration Management, Performance
4. **Phase 4 (Long-term)**: API Design, Monitoring

## Success Metrics
- [ ] All conversions produce detailed logs
- [ ] No file exceeds 500 lines
- [ ] Test coverage > 80%
- [ ] Zero hardcoded configuration values
- [ ] All public APIs have type hints
- [ ] Clear separation of concerns in all modules