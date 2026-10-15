# Optional stages depend on the installed SDK's public catalog.
set(_libpressio_fzgm_probe_template "${CMAKE_CURRENT_LIST_DIR}/check_stage.cpp.in")
function(libpressio_fzgm_check_stage capability stage_type)
  set(probe_dir "${CMAKE_CURRENT_BINARY_DIR}/fzgm-sdk-checks/${capability}")
  configure_file(
    "${_libpressio_fzgm_probe_template}"
    "${probe_dir}/check_stage.cpp" @ONLY)
  # The SDK target also requests CUDA compiler features; probe its headers in C++ only.
  get_target_property(probe_includes FZGMOD::fzgmod INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(shared_includes FZGMOD::fzgmod_compile_settings INTERFACE_INCLUDE_DIRECTORIES)
  list(APPEND probe_includes ${shared_includes} ${CUDAToolkit_INCLUDE_DIRS})
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  # try_compile runs on every configure, so SDK replacements cannot leave stale results.
  try_compile(stage_available "${probe_dir}/build"
    SOURCES "${probe_dir}/check_stage.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${probe_includes}"
    COMPILE_DEFINITIONS -DFZGMOD_BACKEND_CUDA
    CXX_STANDARD 17)
  set(${capability} ${stage_available} PARENT_SCOPE)
endfunction()

libpressio_fzgm_check_stage(LIBPRESSIO_FZGM_HAS_ADM "fz::ADMStage")
libpressio_fzgm_check_stage(LIBPRESSIO_FZGM_HAS_SZP "fz::SZpStage<float>")
libpressio_fzgm_check_stage(LIBPRESSIO_FZGM_HAS_CDF97 "fz::Cdf97Stage<float>")
libpressio_fzgm_check_stage(LIBPRESSIO_FZGM_HAS_SPECK2D "fz::Speck2DStage")
libpressio_fzgm_check_stage(LIBPRESSIO_FZGM_HAS_CDF97_OUTLIER_CORRECT "fz::Cdf97OutlierCorrectStage")
libpressio_fzgm_check_stage(LIBPRESSIO_FZGM_HAS_SZX "fz::SZxStage<float>")
