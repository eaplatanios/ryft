# FFI Conventions

Read this reference when adding, changing, or reviewing raw C interfaces and their Rust wrappers. Apply the shared
[Rust conventions](rust-conventions.md) and [documentation guidelines](documentation-guidelines.md) as relevant.
Crate-specific ownership, context guards, constructors, and invocation macros live in the owning crate's instructions,
including [ryft-mlir](../crates/ryft-mlir/AGENTS.md) and [ryft-pjrt](../crates/ryft-pjrt/AGENTS.md).

- Keep raw FFI details localized and expose safe wrappers by default.
- Use explicit wrapper types around raw handles (e.g., `handle: *mut ...` / `Mlir...` along with `context`/`api`
  fields). Refer to the `ryft-mlir` and `ryft-pjrt` crates for examples.
- For C type wrappers, use the following conversion patterns, as in the `ryft-mlir` and `ryft-pjrt` crates:
  - `unsafe fn from_c_api(...) -> Option<Self>` or `unsafe fn from_c_api(...) -> Result<Self, Error>`, and
  - `unsafe fn to_c_api(&self) -> ...`.
- For opaque C type bindings, follow this exact pattern (including the comments but modifying the struct name):
  ```
  // We represent opaque C types as structs with a particular structure that is following the convention
  // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
  #[repr(C)]
  pub struct PJRT_TopologyDescription {
      _data: [u8; 0],
      _marker: PhantomData<(*mut u8, PhantomPinned)>,
  }
  ```
