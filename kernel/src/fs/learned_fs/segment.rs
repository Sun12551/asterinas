#![expect(dead_code)]

use alloc::{string::String, vec::Vec};
use core::cmp::{max, min, Ordering};
use crate::{
    fs::utils::{InodeMode, InodeType, PageCache}, prelude::*
};
use super::{
    constants::{MAX_MODEL_ERROR, MIN_SEGMENT_SIZE}, fat::{ExfatChain, FatChainFlags}, fs::{LearnedFS, LearnedMountOptions}, inode::Ino, model::SegmentOffset
};

bitflags! {
    pub struct DentryAttr : u16{
        /// This inode is read only.
        const READONLY  = 0x0001;
        /// This inode is hidden. This attribute is not supported in our implementation.
        const HIDDEN    = 0x0002;
        /// This inode belongs to the OS. This attribute is not supported in our implementation.
        const SYSTEM    = 0x0004;
        /// This inode represents a volume. This attribute is not supported in our implementation.
        const VOLUME    = 0x0008;
        /// This inode represents a directory.
        const DIRECTORY = 0x0010;
        /// This file has been touched since the last DOS backup was performed on it. This attribute is not supported in our implementation.
        const ARCHIVE   = 0x0020;
        /// Deleted entry
        const DELETE    = 0x0040;
        /// Spline entry, not used
        const SPLINE    = 0x0080;
    }
}

impl DentryAttr {
    /// Convert attribute bits and a mask to the UNIX mode.
    pub(super) fn make_mode(&self, mount_option: LearnedMountOptions, mode: InodeMode) -> InodeMode {
        let mut ret = mode;
        if self.contains(DentryAttr::READONLY) && !self.contains(DentryAttr::DIRECTORY) {
            ret.remove(InodeMode::S_IWGRP | InodeMode::S_IWUSR | InodeMode::S_IWOTH);
        }
        if self.contains(DentryAttr::DIRECTORY) {
            ret.remove(InodeMode::from_bits_truncate(mount_option.fs_dmask));
        } else {
            ret.remove(InodeMode::from_bits_truncate(mount_option.fs_fmask));
        }
        ret
    }

    pub(super) fn make_type(&self) -> InodeType {
        if self.contains(DentryAttr::DIRECTORY) {
            InodeType::Dir
        } else {
            InodeType::File
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct DirEntry {
    pub(super) name: String,
    pub(super) ino: Ino,
    pub(super) attr: DentryAttr,
}

impl PartialOrd for DirEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.name.cmp(&other.name))
    }
}

impl Ord for DirEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.name.cmp(&other.name)
    }
}

impl DirEntry {
    pub(super) fn new(name: String, ino: Ino, _type: InodeType) -> Self {
        let attr = if _type == InodeType::Dir {
            DentryAttr::DIRECTORY
        } else {
            DentryAttr::empty()
        };
        Self { name, ino, attr }
    }

    pub(super) fn is_deleted(&self) -> bool {
        self.attr.contains(DentryAttr::DELETE)
    }

    pub(super) fn mark_deleted(&mut self) {
        self.attr.insert(DentryAttr::DELETE);
    }

    pub(super) fn is_spline(&self) -> bool {
        self.attr.contains(DentryAttr::SPLINE)
    }

    pub(super) fn mark_spline(&mut self) {
        self.attr.insert(DentryAttr::SPLINE);
    }

    pub(super) fn ino(&self) -> Ino {
        self.ino
    }
}

impl Clone for DirEntry {
    fn clone(&self) -> Self {
        let mut attr = self.attr;
        attr.remove(DentryAttr::SPLINE);
        Self {
            name: self.name.clone(),
            ino: self.ino,
            attr,
        }
    }
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod)]
pub(super) struct DirEntryRaw {
    pub(super) name_offset: u64,
    pub(super) name_length: u64,
    pub(super) ino: u64,
    pub(super) attr: u16,
    pub(super) reserved: [u8; 6],
}

const DIR_ENTRY_SIZE: usize = core::mem::size_of::<DirEntryRaw>();

#[derive(Debug)]
pub(super) struct LearnedSegment {
    pub(super) dentries: Vec<DirEntry>,
    pub(super) size: usize,
    pub(super) total_entry_count: usize,
    pub(super) valid_entry_count: usize,
    start_chain: Arc<ExfatChain>,
    page_cache: PageCache,
    fs: Weak<LearnedFS>,
}

impl LearnedSegment {
    pub(super) fn new(entries: &[DirEntry], fs: Weak<LearnedFS>) -> Arc<RwLock<Self>> {
        // Calculate the size of the segment based on the entries.
        let dentry_size = entries.len() * DIR_ENTRY_SIZE;
        let filename_size = entries
            .iter()
            .map(|entry| entry.name.len() + 1)
            .sum::<usize>();
        let size = dentry_size + filename_size;
        let start_chain = Arc::new(ExfatChain::new(fs.clone(), 0, None, FatChainFlags::ALLOC_POSSIBLE).unwrap());
        let weak_start_chain = Arc::downgrade(&start_chain);
        let segment = Self {
            dentries: entries.to_vec(),
            size,
            total_entry_count: entries.len(),
            valid_entry_count: entries.len(),
            start_chain,
            page_cache: PageCache::with_capacity(size, weak_start_chain).unwrap(),
            fs,
        };
        Arc::new(RwLock::new(segment))
    }

    fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size);
        let mut filename_offset = self.dentries.len() * DIR_ENTRY_SIZE;
        for (i, entry) in self.dentries.iter().enumerate() {
            let name_length = entry.name.len();
            let raw_entry = DirEntryRaw {
                name_offset: filename_offset as u64,
                name_length: name_length as u64,
                ino: entry.ino,
                attr: entry.attr.bits(),
                reserved: [0; 6],
            };
            // Write the raw entry to bytes
            let raw_bytes = raw_entry.as_bytes();
            bytes[i * DIR_ENTRY_SIZE..(i + 1) * DIR_ENTRY_SIZE].copy_from_slice(raw_bytes);
            // Write the filename
            let name_bytes = entry.name.as_bytes();
            bytes[filename_offset..filename_offset + name_length]
                .copy_from_slice(name_bytes);
            // Null-terminate the filename
            bytes[filename_offset + name_length] = b'\0';
            filename_offset += name_length + 1; // +1 for null terminator
        }
        bytes
    }

    fn fs(&self) -> Arc<LearnedFS> {
        self.fs.upgrade().unwrap()
    }

    pub(super) fn merge(&self, merge_entries: &[DirEntry]) -> Vec<Arc<RwLock<Self>>> {
        let tot_cnt = self.valid_entry_count + merge_entries.len();
        let (new_seg_num, avg_cnt, mut rem_cnt) = if tot_cnt < MIN_SEGMENT_SIZE {
            (1, tot_cnt, 0)
        } else {
            let new_seg_num = tot_cnt / MIN_SEGMENT_SIZE;
            let avg_cnt = tot_cnt / new_seg_num;
            let rem_cnt = tot_cnt % new_seg_num;
            (new_seg_num, avg_cnt, rem_cnt)
        };

        let mut new_segments = vec![];
        let mut merge_idx = 0;
        let mut exist_idx = 0;
        for _ in 0..new_seg_num {
            let mut new_dentries = vec![];
            let mut cur_cnt = avg_cnt;
            if rem_cnt > 0 {
                cur_cnt += 1;
                rem_cnt -= 1;
            }
            while new_dentries.len() < cur_cnt {
                while exist_idx < self.dentries.len() && self.dentries[exist_idx].is_deleted() {
                    exist_idx += 1;
                }
                let next_in_self = if merge_idx < merge_entries.len() {
                    exist_idx < self.dentries.len() && self.dentries[exist_idx] < merge_entries[merge_idx]
                } else {
                    exist_idx < self.dentries.len()
                };
                if next_in_self {
                    new_dentries.push(self.dentries[exist_idx].clone());
                    exist_idx += 1;
                } else {
                    new_dentries.push(merge_entries[merge_idx].clone());
                    merge_idx += 1;
                }
            }
            let new_segment = Self::new(&new_dentries, self.fs.clone());
            new_segments.push(new_segment);
        }
        new_segments
    }

    pub(super) fn search_mut(&mut self, predicted: SegmentOffset, target: &str) -> Result<&mut DirEntry> {
        if self.valid_entry_count == 0 {
            return_errno!(Errno::ENOENT);
        }
        let mut left = max(0, predicted as isize - MAX_MODEL_ERROR as isize) as usize;
        let mut right = min(self.valid_entry_count - 1, predicted + MAX_MODEL_ERROR) as usize;
        while left <= right {
            let mid = (left + right) / 2;
            if self.dentries[mid].name == target {
                return Ok(&mut self.dentries[mid]);
            } else if self.dentries[mid].name.as_str() < target {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return_errno!(Errno::ENOENT);
    }

    pub(super) fn search(&self, predicted: SegmentOffset, target: &str) -> Result<&DirEntry> {
        if self.valid_entry_count == 0 {
            return_errno!(Errno::ENOENT);
        }
        let mut left = max(0, predicted as isize - MAX_MODEL_ERROR as isize) as usize;
        let mut right = min(self.valid_entry_count - 1, predicted + MAX_MODEL_ERROR) as usize;
        while left <= right {
            let mid = (left + right) / 2;
            if self.dentries[mid].name == target {
                return Ok(&self.dentries[mid]);
            } else if self.dentries[mid].name.as_str() < target {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return_errno!(Errno::ENOENT);
    }
}