// SPDX-License-Identifier: MPL-2.0

#![expect(dead_code)]
pub(super) const ROOT_INODE_HASH: usize = 1;

// Other pub(super) constants
pub(super) const MAX_CHARSET_SIZE: usize = 6;
pub(super) const MAX_NAME_LENGTH: usize = 8;
pub(super) const MAX_VFSNAME_BUF_SIZE: usize = (MAX_NAME_LENGTH + 1) * MAX_CHARSET_SIZE;

pub(super) const BOOT_SIGNATURE: u16 = 0x6789;
pub(super) const EXBOOT_SIGNATURE: u32 = 0x67890000;
pub(super) const STR_LEARNED: &str = "LEARNED "; // size should be 8

pub(super) const MIN_SEGMENT_SIZE: usize = 32768;
pub(super) const MAX_SEGMENT_SIZE: usize = 2 * MIN_SEGMENT_SIZE;
pub(super) const MAX_BUFFER_SIZE: usize = MIN_SEGMENT_SIZE;

pub(super) const MAX_MODEL_ERROR: usize = 128;
pub(super) const FIXED_POINT_SHIFT: usize = 32;

pub(super) const VOLUME_DIRTY: u16 = 0x0002;
pub(super) const MEDIA_FAILURE: u16 = 0x0004;

// Cluster 0, 1 are reserved, the first cluster is 2 in the cluster heap.
pub(super) const LEARNED_RESERVED_CLUSTERS: u32 = 2;
pub(super) const LEARNED_FIRST_CLUSTER: u32 = 2;

pub(super) const LEARNED_MIN_SECT_SIZE_BITS: u8 = 9;
pub(super) const LEARNED_MAX_SECT_SIZE_BITS: u8 = 12;
