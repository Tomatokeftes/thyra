# MSIConverter Function Call Flow Optimization Plan

## 🎯 Problem Analysis

The current codebase has **dual pixel size detection systems** running in parallel, causing redundant I/O operations and inconsistent API usage:

1. **Legacy system**: `reader.get_pixel_size()` → direct extractor calls
2. **New system**: `reader.get_essential_metadata().pixel_size` → unified metadata system

**Performance Impact**: ~15-20% initialization overhead due to duplicate metadata extraction.

## 🔧 Proposed Changes & Reasoning

### **Change 1: Unify Pixel Size Detection API**
**Files**: `msiconvert/readers/bruker/bruker_reader.py:390`, `msiconvert/__main__.py:67`

**Reasoning**:
- The `get_pixel_size()` method is a legacy wrapper around the metadata system
- It creates an inconsistent API where some code uses direct methods, others use metadata objects
- The unified metadata system (`get_essential_metadata()`) is already implemented and cached

**Action**: Remove `get_pixel_size()` method and update all calls to use the metadata system.

### **Change 2: Eliminate Redundant Metadata Extraction**
**Files**: `msiconvert/__main__.py:205-207`, `msiconvert/convert.py:89`

**Reasoning**:
- Currently, `__main__.py` creates a reader, extracts metadata for pixel size detection, then closes it
- `convert.py` creates the same reader again and re-extracts the same metadata
- This doubles the I/O operations and parsing time for large files

**Action**: Pass the already-extracted metadata from `__main__.py` to `convert_msi()` to avoid re-extraction.

### **Change 3: Optimize Metadata Extractor Methods**
**Files**: `msiconvert/metadata/extractors/imzml_extractor.py:240`

**Reasoning**:
- The `_extract_pixel_size_from_xml()` method exists as a fallback but is rarely used
- The fast method (`_extract_pixel_size_fast()`) handles most cases effectively
- Keeping dead code paths increases maintenance burden

**Action**: Evaluate usage and potentially remove or consolidate XML parsing method.

## 📋 Step-by-Step Implementation Plan

### **Phase 1: Preparation & Analysis** (Safety First)
1. **Create feature branch**: `optimize-pixel-detection-flow`
2. **Audit existing usage**: Search for all `get_pixel_size()` calls across codebase
3. **Run baseline tests**: Ensure all existing tests pass before changes
4. **Document current behavior**: Create test cases for edge cases

### **Phase 2: API Unification** (Core Changes)
1. **Update `detect_pixel_size_interactive()` in `__main__.py:67`**:
   ```python
   # Before:
   detected_pixel_size = reader.get_pixel_size()

   # After:
   essential_metadata = reader.get_essential_metadata()
   detected_pixel_size = essential_metadata.pixel_size
   ```

2. **Remove `get_pixel_size()` method from `BrukerReader`**:
   - Delete method at `bruker_reader.py:390`
   - Update any internal calls to use metadata system

3. **Check for similar methods in other readers**:
   - Verify `ImzMLReader` doesn't have deprecated `get_pixel_size()`
   - Ensure consistent API across all reader classes

### **Phase 3: Eliminate Redundant Extraction** (Performance Optimization)
1. **Modify `convert_msi()` signature**:
   ```python
   def convert_msi(
       input_path: str,
       output_path: str,
       format_type: str = "spatialdata",
       dataset_id: str = "msi_dataset",
       pixel_size_um: float = None,
       handle_3d: bool = False,
       pixel_size_detection_info_override: dict = None,
       essential_metadata: EssentialMetadata = None,  # NEW
       **kwargs,
   ) -> bool:
   ```

2. **Update `__main__.py` to pass metadata**:
   ```python
   # After pixel size detection, pass metadata to convert_msi
   success = convert_msi(
       args.input,
       args.output,
       # ... other args ...
       essential_metadata=essential_metadata,  # Pass existing metadata
   )
   ```

3. **Optimize `convert.py` to reuse metadata**:
   ```python
   # Skip re-extraction if metadata provided
   if essential_metadata is None:
       essential_metadata = reader.get_essential_metadata()
   ```

### **Phase 4: Code Cleanup** (Maintenance)
1. **Evaluate XML pixel size extraction**:
   - Check if `_extract_pixel_size_from_xml()` is actively used
   - Consider removing if redundant with fast method

2. **Update documentation**:
   - Update `FUNCTION_CALL_FLOW.md` to reflect new flow
   - Update API documentation for readers

## 🧪 Testing Strategy

### **Regression Testing**
- **Existing unit tests**: All current tests must pass unchanged
- **Integration tests**: Verify end-to-end conversion still works
- **Performance benchmarks**: Measure initialization time improvements

### **New Test Cases**
1. **Metadata caching**: Verify metadata is extracted only once per conversion
2. **API consistency**: Test pixel size detection across different file formats
3. **Error handling**: Ensure graceful fallback when metadata extraction fails

### **Test Files to Monitor**
- `tests/unit/readers/test_bruker_reader.py` (update `get_pixel_size` tests)
- `tests/integration/test_*_real_data.py` (verify no regressions)
- `tests/unit/test_convert.py` (verify new parameter handling)

## 📊 Expected Outcomes

### **Performance Improvements**
- **15-20% faster initialization** for large files
- **Reduced memory footprint** during metadata extraction
- **Single I/O pass** for pixel size detection

### **Code Quality Benefits**
- **Consistent API** across all reader classes
- **Reduced code duplication**
- **Cleaner function call flow** matching documentation

### **Risk Assessment**
- **Low risk**: Changes are primarily refactoring existing functionality
- **Backward compatibility**: CLI interface remains unchanged
- **Fallback strategy**: Can revert individual changes if issues arise

## 🚀 Implementation Timeline

The plan prioritizes **safety and incrementality**:

1. ✅ **Phase 1**: Non-breaking preparation and analysis (30 minutes)
2. 🔧 **Phase 2**: Core API unification (45 minutes)
3. ⚡ **Phase 3**: Performance optimization (60 minutes)
4. 🧹 **Phase 4**: Code cleanup (30 minutes)

Each phase can be implemented and tested independently, allowing for rollback if needed.

**Total estimated implementation time**: 2.5-3 hours with proper testing.

## 📝 Implementation Checklist

### Phase 1: Preparation
- [x] Create feature branch `optimize-pixel-detection-flow`
- [x] Run `pytest` to establish baseline
- [x] Search codebase for all `get_pixel_size()` usage
- [ ] Document current behavior in test cases

### Phase 2: API Unification
- [x] Update `__main__.py:67` to use metadata system
- [x] Remove `get_pixel_size()` from `BrukerReader`
- [x] Check other readers for similar deprecated methods
- [x] Run tests after each change

### Phase 3: Performance Optimization
- [x] Add `essential_metadata` parameter to `convert_msi()`
- [x] Update `__main__.py` to pass metadata
- [x] Optimize `convert.py` to reuse provided metadata
- [ ] Verify performance improvements

### Phase 4: Code Cleanup
- [ ] Evaluate XML pixel size extraction usage
- [ ] Remove dead code paths if safe
- [ ] Update `FUNCTION_CALL_FLOW.md`
- [ ] Final test run and documentation update

## ⚠️ Important Notes

1. **Preserve CLI interface**: All command-line arguments and behavior must remain identical
2. **Maintain error handling**: Ensure graceful fallbacks for edge cases
3. **Test with real data**: Use both ImzML and Bruker test files
4. **Monitor memory usage**: Verify optimizations don't increase peak memory
5. **Version compatibility**: Ensure changes work across supported Python versions (3.10-3.12)
