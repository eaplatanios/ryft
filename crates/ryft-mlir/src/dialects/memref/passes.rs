use std::sync::OnceLock;

use crate::mlir_pass;

mlir_pass!(expand_strided_metadata_pass, ExpandStridedMetadataPass);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_strided_metadata_pass() {
        // Verify that there are no segmentation faults, even when we try to register multiple times.
        register_expand_strided_metadata_pass();
        register_expand_strided_metadata_pass();
        let _ = create_expand_strided_metadata_pass().unwrap();
    }
}
