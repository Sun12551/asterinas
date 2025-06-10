// SPDX-License-Identifier: MPL-2.0

#![expect(dead_code)]
#![expect(unused_variables)]

use core::ops::Range;

use align_ext::AlignExt;
use bitvec::prelude::*;

use super::{
    constants::LEARNED_RESERVED_CLUSTERS,
    fat::ClusterID,
    fs::LearnedFS,
};
use crate::prelude::*;

// TODO: use u64
type BitStore = u8;

const BITS_PER_BYTE: usize = 8;

#[derive(Debug, Default)]
pub(super) struct LearnedBitmap {
    // Start offset of the bitmap
    start_offset: usize,
    bitvec: BitVec<BitStore>,
    dirty_bytes: VecDeque<Range<usize>>,

    // Used to track the number of free clusters.
    num_free_cluster: u32,
    fs: Weak<LearnedFS>,
}

impl LearnedBitmap {
    pub(super) fn load(
        fs_weak: Weak<LearnedFS>,
    ) -> Result<Self> {
        let fs = fs_weak.upgrade().unwrap();
        let super_block = fs.super_block();
        let mut buf = vec![0; super_block.num_bitmap_sectors as usize * super_block.sector_size as usize];
        let start_offset = super_block.bitmap_start_sector as usize * super_block.sector_size as usize;
        fs.read_meta_at(start_offset, &mut buf)?;
        let mut free_cluster_num = 0;
        for idx in 0..fs.super_block().num_clusters - LEARNED_RESERVED_CLUSTERS {
            if (buf[idx as usize / BITS_PER_BYTE] & (1 << (idx % BITS_PER_BYTE as u32))) == 0 {
                free_cluster_num += 1;
            }
        }
        Ok(LearnedBitmap {
            start_offset: start_offset,
            bitvec: BitVec::from_slice(&buf),
            dirty_bytes: VecDeque::new(),
            num_free_cluster: free_cluster_num,
            fs: fs_weak,
        })
    }

    fn fs(&self) -> Arc<LearnedFS> {
        self.fs.upgrade().unwrap()
    }

    fn is_used(&self, bit: usize) -> bool {
        *(self.bitvec.get(bit).unwrap())
    }

    pub(super) fn set_used(&mut self, cluster: u32, sync: bool) -> Result<()> {
        self.set_range(cluster..cluster + 1, true, sync)
    }

    pub(super) fn set_unused(&mut self, cluster: u32, sync: bool) -> Result<()> {
        self.set_range(cluster..cluster + 1, false, sync)
    }

    pub(super) fn set_range_used(&mut self, clusters: Range<ClusterID>, sync: bool) -> Result<()> {
        self.set_range(clusters, true, sync)
    }

    pub(super) fn set_range_unused(
        &mut self,
        clusters: Range<ClusterID>,
        sync: bool,
    ) -> Result<()> {
        self.set_range(clusters, false, sync)
    }

    pub(super) fn is_cluster_unused(&self, cluster: u32) -> Result<bool> {
        self.is_cluster_range_unused(cluster..cluster + 1)
    }

    pub(super) fn is_cluster_range_unused(&self, clusters: Range<ClusterID>) -> Result<bool> {
        if !self.fs().is_cluster_range_valid(clusters.clone()) {
            return_errno_with_message!(Errno::EINVAL, "invalid cluster ranges.")
        }

        for id in clusters {
            if self.bitvec[(id - LEARNED_RESERVED_CLUSTERS) as usize] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return the first unused cluster.
    pub(super) fn find_next_unused_cluster(&self, cluster: ClusterID) -> Result<ClusterID> {
        let clusters = self.find_next_unused_cluster_range_by_bits(cluster, 1)?;
        Ok(clusters.start)
    }

    /// Return the first unused cluster range, set num_clusters=1 to find a single cluster.
    fn find_next_unused_cluster_range_by_bits(
        &self,
        search_start_cluster: ClusterID,
        num_clusters: u32,
    ) -> Result<Range<ClusterID>> {
        if !self
            .fs()
            .is_cluster_range_valid(search_start_cluster..search_start_cluster + num_clusters)
        {
            return_errno_with_message!(Errno::ENOSPC, "free contiguous clusters not available.")
        }

        let mut cur_index = search_start_cluster - LEARNED_RESERVED_CLUSTERS;
        let end_index = self.fs().super_block().num_clusters - LEARNED_RESERVED_CLUSTERS;
        let search_end_index = end_index - num_clusters + 1;
        let mut range_start_index: ClusterID;

        while cur_index < search_end_index {
            if !self.is_used(cur_index as usize) {
                range_start_index = cur_index;
                let mut cnt = 0;
                while cnt < num_clusters
                    && cur_index < end_index
                    && !self.is_used(cur_index as usize)
                {
                    cnt += 1;
                    cur_index += 1;
                }
                if cnt >= num_clusters {
                    return Ok(range_start_index + LEARNED_RESERVED_CLUSTERS
                        ..range_start_index + LEARNED_RESERVED_CLUSTERS + num_clusters);
                }
            }
            cur_index += 1;
        }
        return_errno!(Errno::ENOSPC)
    }

    /// Make sure the bit at the range start position is 0.
    fn adjust_head_pos(
        &self,
        bytes: &[BitStore],
        mut cur_unit_index: u32,
        mut cur_unit_offset: u32,
        total_cluster_num: u32,
    ) -> (u32, u32) {
        let unit_size: u32 = (BITS_PER_BYTE * core::mem::size_of::<BitStore>()) as u32;
        while cur_unit_index < total_cluster_num {
            let leading_zeros = bytes[cur_unit_index as usize].leading_zeros();
            let head_cluster_num = unit_size - cur_unit_offset;
            if leading_zeros == 0 {
                // Fall over to the next unit, we need to continue checking.
                cur_unit_index += 1;
                cur_unit_offset = 0;
            } else {
                // Stop at current unit, we may need to adjust the cur_offset
                cur_unit_offset = cur_unit_offset.max(unit_size - leading_zeros);
                break;
            }
        }
        (cur_unit_index, cur_unit_offset)
    }

    /// Check if the next mid_unit_num units are zero.
    /// If not, return the index of the first not zero unit.
    fn check_mid_units(&self, bytes: &[BitStore], cur_unit_index: u32, mid_unit_num: u32) -> u32 {
        for i in 1..mid_unit_num + 1 {
            if bytes[(cur_unit_index + i) as usize] != 0 {
                return cur_unit_index + 1;
            }
        }
        cur_unit_index
    }

    /// Check if the tail unit is valid.
    /// Currently not used.
    fn check_tail_bits(
        &self,
        bytes: &[BitStore],
        tail_idx: u32,
        tail_cluster_num: u32,
        complete_unit_num: u32,
        rest_cluster_num: u32,
    ) -> bool {
        let valid_bytes_num = if rest_cluster_num > 0 {
            complete_unit_num + 1
        } else {
            complete_unit_num
        };
        let mut tail_byte: u8 = 0;
        if tail_idx == complete_unit_num {
            tail_byte |= 0xFF_u8 - ((1_u8 << rest_cluster_num) - 1);
        }
        if tail_idx < valid_bytes_num {
            tail_byte |= bytes[tail_idx as usize];
        }
        let tailing_zeros = tail_byte.trailing_zeros();
        tailing_zeros >= tail_cluster_num
    }

    fn make_range(
        &self,
        cur_unit_index: u32,
        cur_unit_offset: u32,
        num_clusters: u32,
    ) -> Range<ClusterID> {
        let unit_size: u32 = (BITS_PER_BYTE * core::mem::size_of::<BitStore>()) as u32;
        let result_bit_index = cur_unit_index * unit_size + cur_unit_offset;
        result_bit_index + LEARNED_RESERVED_CLUSTERS
            ..result_bit_index + LEARNED_RESERVED_CLUSTERS + num_clusters
    }

    /// Return the next contiguous unused clusters, set cluster_num=1 to find a single cluster
    pub(super) fn find_next_unused_cluster_range(
        &self,
        search_start_cluster: ClusterID,
        num_clusters: u32,
    ) -> Result<Range<ClusterID>> {
        if !self
            .fs()
            .is_cluster_range_valid(search_start_cluster..search_start_cluster + num_clusters)
        {
            return_errno!(Errno::ENOSPC)
        }

        let bytes: &[BitStore] = self.bitvec.as_raw_slice();
        let unit_size: u32 = (BITS_PER_BYTE * core::mem::size_of::<BitStore>()) as u32;
        let start_cluster_index = search_start_cluster - LEARNED_RESERVED_CLUSTERS;
        let mut cur_unit_index = start_cluster_index / unit_size;
        let mut cur_unit_offset = start_cluster_index % unit_size;
        let total_cluster_num = self.fs().super_block().num_clusters - LEARNED_RESERVED_CLUSTERS;
        let complete_unit_num = total_cluster_num / unit_size;
        let rest_cluster_num = total_cluster_num % unit_size;
        let valid_bytes_num = if rest_cluster_num > 0 {
            complete_unit_num + 1
        } else {
            complete_unit_num
        };
        if num_clusters <= unit_size {
            // If this case, back to the simple function
            return self.find_next_unused_cluster_range_by_bits(search_start_cluster, num_clusters);
        }
        // Treat a continuous bit chunk as lead_bits+mid_units+tail_bits (mid_units are unit aligned)
        // For example: 11110000 00000000 00000000 00111111
        //                  **** -------- -------- ..
        //                  ^(start bit)
        // (*): head_bits;  (-): mid_units;  (.): tail_bits
        // The start bit can be identified with a pair (cur_unit_index, cur_unit_offset)
        while cur_unit_index < complete_unit_num {
            // First, adjust the cur_idx to a proper head.
            (cur_unit_index, cur_unit_offset) =
                self.adjust_head_pos(bytes, cur_unit_index, cur_unit_offset, total_cluster_num);
            let head_cluster_num = unit_size - cur_unit_offset;
            let mid_unit_num = (num_clusters - head_cluster_num) / unit_size;
            let tail_cluster_num = (num_clusters - head_cluster_num) % unit_size;
            // If the last complete unit to be check is out of range, stop searching
            if cur_unit_index + mid_unit_num >= complete_unit_num {
                break;
            }
            // Then check for the mid units, these units should be all zero
            // Due to previous check, there will be no array out of bounds situation
            let ret = self.check_mid_units(bytes, cur_unit_index, mid_unit_num);
            if ret != cur_unit_index {
                // Mid_checks failed, should go back to the first step.
                cur_unit_index = ret;
                cur_unit_offset = 0;
                continue;
            }
            // At last, check for the tail bits
            if tail_cluster_num == 0 {
                return Ok(self.make_range(cur_unit_index, cur_unit_offset, num_clusters));
            }
            let mut tail_byte: u8 = 0;
            let tail_idx = cur_unit_index + mid_unit_num + 1;
            if tail_idx == complete_unit_num {
                tail_byte |= 0xFF_u8 - ((1_u8 << rest_cluster_num) - 1);
            }
            if tail_idx < valid_bytes_num {
                tail_byte |= bytes[tail_idx as usize];
            }
            let tailing_zeros = tail_byte.trailing_zeros();
            if tail_cluster_num > tailing_zeros {
                cur_unit_index = tail_idx;
                cur_unit_offset = tailing_zeros + 1;
                continue;
            }
            // If we reach here, it means we have found a result
            return Ok(self.make_range(cur_unit_index, cur_unit_offset, num_clusters));
        }
        return_errno!(Errno::ENOSPC)
    }

    pub(super) fn num_free_clusters(&self) -> u32 {
        self.num_free_cluster
    }

    fn set_range(&mut self, clusters: Range<ClusterID>, bit: bool, sync: bool) -> Result<()> {
        if !self.fs().is_cluster_range_valid(clusters.clone()) {
            return_errno_with_message!(Errno::EINVAL, "invalid cluster ranges.")
        }

        for cluster_id in clusters.clone() {
            let index = (cluster_id - LEARNED_RESERVED_CLUSTERS) as usize;
            let old_bit = self.is_used(index);
            self.bitvec.set(index, bit);

            if !old_bit && bit {
                self.num_free_cluster -= 1;
            } else if old_bit && !bit {
                self.num_free_cluster += 1;
            }
        }

        self.write_to_disk(clusters.clone(), sync)?;

        Ok(())
    }

    fn write_to_disk(&mut self, clusters: Range<ClusterID>, sync: bool) -> Result<()> {
        let unit_size = core::mem::size_of::<BitStore>() * BITS_PER_BYTE;
        let start_byte_off: usize = (clusters.start - LEARNED_RESERVED_CLUSTERS) as usize / unit_size;
        let end_byte_off: usize =
            ((clusters.end - LEARNED_RESERVED_CLUSTERS) as usize).align_up(unit_size) / unit_size;

        let bytes: &[BitStore] = self.bitvec.as_raw_slice();
        let byte_chunk = &bytes[start_byte_off..end_byte_off];

        let phys_offset = self.start_offset + start_byte_off * unit_size;
        self.fs().write_meta_at(phys_offset, byte_chunk)?;

        let byte_range = phys_offset..phys_offset + byte_chunk.len();

        if sync {
            self.fs().sync_meta_at(byte_range.clone())?;
        } else {
            self.dirty_bytes.push_back(byte_range.clone());
        }

        Ok(())
    }

    pub(super) fn sync(&mut self) -> Result<()> {
        while let Some(range) = self.dirty_bytes.pop_front() {
            self.fs().sync_meta_at(range)?;
        }
        Ok(())
    }
}
