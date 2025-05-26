#![expect(dead_code)]

use alloc::{string::String, vec::Vec};
use core::cmp::{max, min, Ordering};
use crate::prelude::*;
use super::{
    model::SegmentOffset,
    inode::{Ino, DELETE_FLAG, INO_MASK},
    constants::{MIN_SEGMENT_SIZE, MAX_MODEL_ERROR},
};


#[derive(Debug, PartialEq, Eq)]
pub(super) struct DirEntry {
    pub(super) name: String,
    pub(super) ino: Ino,
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
    pub(super) fn new(name: String, ino: Ino) -> Self {
        Self { name, ino }
    }

    pub(super) fn is_deleted(&self) -> bool {
        self.ino & DELETE_FLAG != 0
    }

    pub(super) fn mark_deleted(&mut self) {
        self.ino |= DELETE_FLAG;
    }

    pub(super) fn ino(&self) -> Ino {
        self.ino & INO_MASK
    }
}

impl Clone for DirEntry {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            ino: self.ino & INO_MASK,
        }
    }
}

#[derive(Debug)]
pub(super) struct LearnedSegment {
    pub(super) dentries: Vec<DirEntry>,
    pub(super) valid_entry_count: usize,
}

impl LearnedSegment {
    pub(super) fn new(entries: &[DirEntry]) -> Self {
        Self {
            dentries: entries.to_vec(),
            valid_entry_count: entries.len(),
        }
    }

    pub(super) fn push(&mut self, entry: DirEntry) {
        self.dentries.push(entry);
        self.valid_entry_count += 1;
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
            let mut new_segment = Self::new(&[]);
            let mut cur_cnt = avg_cnt;
            if rem_cnt > 0 {
                cur_cnt += 1;
                rem_cnt -= 1;
            }
            while new_segment.valid_entry_count < cur_cnt {
                while exist_idx < self.dentries.len() && self.dentries[exist_idx].is_deleted() {
                    exist_idx += 1;
                }
                let next_in_self = if merge_idx < merge_entries.len() {
                    exist_idx < self.dentries.len() && self.dentries[exist_idx] < merge_entries[merge_idx]
                } else {
                    exist_idx < self.dentries.len()
                };
                if next_in_self {
                    new_segment.dentries.push(self.dentries[exist_idx].clone());
                    exist_idx += 1;
                } else {
                    new_segment.dentries.push(merge_entries[merge_idx].clone());
                    merge_idx += 1;
                }
            }
            new_segment.valid_entry_count = new_segment.dentries.len();
            new_segments.push(Arc::new(RwLock::new(new_segment)));
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