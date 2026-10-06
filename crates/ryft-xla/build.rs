//! Build script for `ryft-xla`, which only adjusts how this crate's own executables (e.g., its unit test binary)
//! are linked on macOS.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // The unit test binary of this crate is very large, and under the `v0` symbol mangling scheme that recent Rust
    // toolchains use by default (e.g., 1.99.0), which spells out every generic argument in each symbol name, its local
    // symbols alone take more than a gigabyte. macOS loads a main executable at `0x100000000` and maps the dyld shared
    // cache at `0x180000000`, so an executable larger than 2 GiB overlaps the cache, which then fails to map and takes
    // every system library (e.g., `libSystem` and `libiconv`) with it. Linking with `-x` drops the non-global symbols,
    // which nothing needs at runtime. Backtraces and debuggers still resolve function names through the debug map,
    // which `-x` keeps, so the flag is only applied when debug information is enabled, leaving the symbol tables of
    // builds without debug information (e.g., optimized builds that are profiled) intact. Build script link arguments
    // only apply to this crate's own executables, never to the executables of crates that depend on it.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let debug = std::env::var("DEBUG").is_ok_and(|debug| debug != "false" && debug != "0" && debug != "none");
    if target_os == "macos" && debug {
        println!("cargo:rustc-link-arg=-Wl,-x");
    }
}
