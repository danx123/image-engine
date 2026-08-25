# image_engine

A Rust + [PyO3](https://pyo3.rs/) native extension that mirrors a focused subset of the OpenCV Python API (`cv2`), built as a drop-in performance backend for **Macan Image Viewer** and **Macan Efek**. It wraps [opencv-rust](https://github.com/twistedfall/opencv-rust) behind a minimal, `cv2`-shaped surface so existing Python code can swap `cv2` calls for `image_engine` calls with little to no logic changes, while getting compiled-Rust performance and a self-contained wheel (no separate OpenCV install required on the end user's machine).

## Why this exists

- **Performance** — image and video-metadata operations run as compiled Rust/OpenCV instead of interpreted Python + `cv2`'s Python bindings overhead.
- **Minimal surface, minimal binary** — only the OpenCV modules actually used (`core`, `imgproc`, `imgcodecs`, `photo`, `videoio`) are built in, keeping the compiled OpenCV and the resulting wheel small.
- **Drop-in familiarity** — function names, signatures, and constants closely follow `cv2.*`, so porting call sites from `cv2` is mostly mechanical.
- **Self-contained distribution** — the Windows wheel bundles the OpenCV DLL via `delvewheel`, so it works on machines without OpenCV installed.

## API Overview

### Core `cv2`-equivalent functions
`imread`, `imwrite`, `cvt_color`, `resize`, `rotate`, `flip`, `add_weighted`

### Effects / image processing (mirrors `cv2.*`)
`gaussian_blur`, `filter_2d`, `bilateral_filter`, `median_blur`, `apply_color_map`, `convert_scale_abs`, `lut`, `split`, `merge`, `bitwise_not`, `bitwise_and`, `add_scalar`, `subtract_scalar`, `divide`, `transform`, `hconcat`, `vconcat`, `canny`, `adaptive_threshold`

### Higher-level effects (ported from Python, now native)
- `manual_grayscale` — custom grayscale conversion
- `apply_sepia` — sepia tone via a 3×3 color transform matrix, alpha-channel-safe
- `adjust_gamma` — gamma correction via LUT
- `adjust_brightness_contrast` — brightness/contrast in one call (`convertScaleAbs`)
- `adjust_channel_mixer` — independent R/G/B channel shifts
- `adjust_saturation` / `adjust_hue` — HSV-space adjustments
- `apply_vignette` — Gaussian-based vignette darkening (RGB channels only; alpha untouched)
- `apply_sharpen` — standard 3×3 sharpen kernel
- `apply_unsharp_mask` — Gaussian-blur-based unsharp masking
- `equalize_hist` — histogram equalization (luma-only for color images, via YCrCb, to avoid color shifts)

### NumPy bridge
- `numpy_to_mat(array: np.ndarray) -> PyMat` — converts a contiguous `(H, W, C)` `uint8` array into an OpenCV `Mat`
- `mat_to_numpy(mat: PyMat) -> np.ndarray` — converts back to NumPy

This lets `macan_efek.py` keep working with `np.ndarray` for things like `QImage` conversion, PIL interop, or canvas slicing, without depending on `cv2` at all.

### `VideoCapture`
A metadata-only wrapper around `cv2.VideoCapture` (`is_opened`, `get`, `set`, `release`, context-manager support) for reading properties like FPS, resolution, frame count, and FOURCC — **not** used for frame decoding.

### Constants
All the `cv2.*` constants needed by the functions above are re-exported with matching names: `IMREAD_*`, `IMWRITE_*`, `COLOR_*`, `INTER_*`, `ROTATE_*`, `COLORMAP_*`, `BORDER_*`, `ADAPTIVE_THRESH_*` / `THRESH_*`, and `CAP_PROP_*` / `CAP_ANY`.

## Architecture Notes

- **`PyMat`** wraps `opencv::core::Mat` as a `#[pyclass(unsendable)]` — `Mat` isn't a native PyO3 type and wraps a non-`Sync` raw pointer, so it can only be accessed from the Python thread that created it. All functions operate on `PyMat` at the Python boundary and unwrap to `&Mat` internally.
- **`EngineError`** is a local error type bridging `opencv::Error` and `pyo3::PyErr` — Rust's orphan rule blocks a direct `impl From<opencv::Error> for PyErr` since both types are foreign to this crate, so `EngineError` sits in between and converts cleanly to `PyErr` for use with `?`.
- **NumPy/ndarray version pinning**: the crate deliberately does **not** declare its own `ndarray` dependency. The `numpy` crate supports a semver-incompatible range of `ndarray` versions (0.15–0.17); declaring `ndarray` separately risks Cargo resolving two different major versions, which would make `Array3<u8>` incompatible with the `IntoPyArray` trait `numpy` expects. `lib.rs` always uses `numpy::ndarray::Array3` (the re-export) instead of importing `ndarray` directly.
- **OpenCV version resolution**: as of `opencv-rust` ≥ 0.53.0, there's no `opencv-4XX` Cargo feature to select an OpenCV version, and `buildtime_bindgen` has been removed (bindings are pre-generated and shipped in the crate). The OpenCV version is instead pinned by whichever OpenCV build CI produces (see `build.yml`), discovered automatically by header inspection at build time.

## Requirements

- Rust (stable toolchain)
- Python 3.9+ (built with `abi3-py39` — one wheel works across Python 3.9+)
- [Maturin](https://www.maturin.rs/) for building
- OpenCV 4.8.0, built with the specific module set this crate needs (see [Build](#build) below) — **not** a generic system OpenCV install

## Cargo Features / Dependencies

| Crate | Purpose |
|---|---|
| `pyo3` (`extension-module`, `abi3-py39`) | Python bindings, ABI3-stable across Python 3.9+ |
| `numpy` | NumPy `ndarray` bridge (`IntoPyArray`, `PyReadonlyArray3`) |
| `opencv` (`default-features = false`) | Only `imgproc`, `imgcodecs`, `photo`, `videoio` modules enabled — kept minimal on purpose |
| `rayon` | Available for future parallelization; not currently required by any exposed function |

`opencv`'s `core` module is always included and can't be toggled — declaring it as a feature is a build error on opencv-rust ≥ 0.53.0.

## Build

Building `image_engine` requires OpenCV compiled locally (or use the CI-produced wheel from GitHub Actions — see below). The general local workflow:

```bash
maturin develop --release
# or, for a distributable wheel:
maturin build --release --interpreter python
```

You'll need OpenCV's headers/libs discoverable via one of `opencv-rust`'s probe mechanisms (`pkg_config`, `cmake`, `vcpkg`, or manual `OPENCV_INCLUDE_PATHS` / `OPENCV_LINK_PATHS` / `OPENCV_LINK_LIBS` environment variables). On Windows, the manual "environment" probe is the most reliable — CMake's `--find-package` probe has known issues on Windows (`CMAKE_SIZEOF_VOID_P is not defined`).

### CI (GitHub Actions — `build.yml`)

The included workflow builds a Windows x64 wheel for Python 3.13 end-to-end:

1. **Cache OpenCV** — an OpenCV 4.8.0 build (module-minimal, `BUILD_opencv_world=ON`) is cached by OS + version + workflow-file hash, so most runs skip rebuilding OpenCV entirely (~500MB+ source clone avoided on cache hit).
2. **Build OpenCV** (on cache miss) — CMake configures OpenCV with only `core`, `imgproc`, `imgcodecs`, `photo`, `videoio` enabled, all GUI/backends/acceleration (Qt, GTK, FFmpeg, CUDA, OpenCL, IPP, etc.) turned off, and `BUILD_opencv_world=ON` so everything ships as a single `opencv_world480.dll`/`.lib`.
3. **Locate OpenCV artifacts** — the built `.lib`/`.dll`/include paths are discovered and exported as `OPENCV_INCLUDE_PATHS` / `OPENCV_LINK_PATHS` / `OPENCV_LINK_LIBS` / `OPENCV_DLL_DIR`, with `OPENCV_DISABLE_PROBES` forcing `opencv-rust` to use only the manual environment probe (avoiding the fragile CMake auto-discovery path on Windows).
4. **Build the wheel** — `maturin build --release --interpreter python`, with `OPENCV_MSVC_CRT=dynamic`.
5. **Repair the wheel** — since OpenCV is linked dynamically, `delvewheel` bundles `opencv_world480.dll` into the wheel so it works on machines without OpenCV installed.
6. **Release** — on version tags (`v*`), the built wheel is attached to the corresponding GitHub Release.

Rust build artifacts are cached via `Swatinem/rust-cache` (smarter than manual `actions/cache` — keys automatically incorporate rustc version, target triple, and `Cargo.lock`).

## Usage from Python

```python
import image_engine as ie

img = ie.imread("photo.jpg")
gray = ie.cvt_color(img, ie.COLOR_BGR2GRAY)
blurred = ie.gaussian_blur(img, (5, 5), 0)
sepia = ie.apply_sepia(img)
ie.imwrite("out.jpg", sepia)

# NumPy interop
import numpy as np
arr = np.ascontiguousarray(some_hwc_uint8_array)
mat = ie.numpy_to_mat(arr)
back_to_numpy = ie.mat_to_numpy(mat)

# Video metadata (not frame decoding)
with ie.VideoCapture("clip.mp4") as cap:
    fps = cap.get(ie.CAP_PROP_FPS)
    frames = cap.get(ie.CAP_PROP_FRAME_COUNT)
```

## Scope / Limitations

- This is **not** a general-purpose OpenCV binding — only the functions and modules needed by Macan Image Viewer / Macan Efek are exposed.
- `VideoCapture` is metadata-only (`get`/`set`/`is_opened`); it does not decode or return video frames. Frame decoding is handled elsewhere in the Macan suite (see `media_engine`).
- Arrays passed to `numpy_to_mat` must be contiguous `(H, W, C)` `uint8`; non-contiguous arrays need `np.ascontiguousarray()` first.
- `PyMat` instances are `unsendable` — they can only be used from the Python thread that created them, not shared across threads.
