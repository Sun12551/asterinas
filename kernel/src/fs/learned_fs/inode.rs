// SPDX-License-Identifier: MPL-2.0

#![expect(unused_variables)]

use alloc::string::String;
use core::{cmp::Ordering, time::Duration};

pub(super) use align_ext::AlignExt;
use aster_block::{
    bio::{BioDirection, BioSegment, BioWaiter},
    id::{Bid, BlockId},
    BLOCK_SIZE,
};
use aster_rights::Full;
use ostd::mm::{Segment, VmIo};
use alloc::collections::LinkedList;
use hashbrown::HashMap;

use super::{
    constants::*,
    fat::{ClusterAllocator, ClusterID, ExfatChainPosition, FatChainFlags, ExfatChain},
    fs::{LEARNED_ROOT_INO, LearnedFS},
    model::Model,
    segment::{DirEntry, LearnedSegment, DentryAttr},
    utils::DosTimestamp,
};
use crate::{
    events::IoEvents,
    fs::{
        path::{is_dot, is_dot_or_dotdot, is_dotdot},
        utils::{
            CachePage, DirentVisitor, Extension, Inode, InodeMode, InodeType, IoctlCmd, Metadata,
            MknodType, PageCache, PageCacheBackend,
        },
    },
    prelude::*,
    process::{signal::PollHandle, Gid, Uid},
    vm::vmo::Vmo,
};

///Inode number
pub type Ino = u64;

#[repr(C, packed)]
#[derive(Debug, Default, Clone, Copy, Pod)]
pub(super) struct LearnedInodeOnDisk {
    pub(super) valid_size: u64,
    pub(super) start_cluster: u32,
    pub(super) reserved: u16,
    pub(super) create_time: u16,
    pub(super) create_date: u16,
    pub(super) modify_time: u16,
    pub(super) modify_date: u16,
    pub(super) access_time: u16,
    pub(super) access_date: u16,
    pub(super) create_time_cs: u8,
    pub(super) modify_time_cs: u8,
    pub(super) create_utc_offset: u8,
    pub(super) modify_utc_offset: u8,
    pub(super) access_utc_offset: u8,
    pub(super) flags: u8, // bit0: AllocationPossible (must be 1); bit1: NoFatChain (=1 <=> contiguous)
}

pub(super) const LEARNED_INODE_ONDISK_SIZE: usize = 32;

impl LearnedInodeOnDisk {
    pub(super) fn new(fs: Arc<LearnedFS>) -> Result<Self> {
        let dos_time = DosTimestamp::now()?;

        Ok(Self {
            valid_size: 0,
            start_cluster: 0,
            reserved: 0,
            create_time: dos_time.time,
            create_date: dos_time.date,
            modify_time: dos_time.time,
            modify_date: dos_time.date,
            access_time: dos_time.time,
            access_date: dos_time.date,
            create_time_cs: dos_time.increment_10ms,
            modify_time_cs: dos_time.increment_10ms,
            create_utc_offset: dos_time.utc_offset,
            modify_utc_offset: dos_time.utc_offset,
            access_utc_offset: dos_time.utc_offset,
            flags: FatChainFlags::FAT_CHAIN_NOT_IN_USE.bits(),
        })
    }
}

#[derive(Debug)]
pub struct LearnedInode {
    inner: RwMutex<LearnedInodeInner>,
    extension: Extension,
}

#[derive(Debug)]
struct LearnedInodeInner {
    /// Inode number.
    ino: Ino,

    /// Dentry set position in its parent directory.
    dentry_set_position: ExfatChainPosition,
    /// Dentry set size in bytes.
    dentry_set_size: usize,
    /// The entry number of the dentry.
    dentry_entry: u32,
    /// Inode type, File or Dir.
    inode_type: InodeType,

    attr: DentryAttr,

    /// Start position on disk, this is undefined if the allocated size is 0.
    start_chain: ExfatChain,

    /// Valid size of the file.
    size: usize,
    /// Allocated size, for directory, size is always equal to size_allocated.
    size_allocated: usize,

    /// Access time, updated after reading.
    atime: DosTimestamp,
    /// Modification time, updated only on write.
    mtime: DosTimestamp,
    /// Creation time.
    ctime: DosTimestamp,

    /// Number of sub inodes.
    num_sub_inodes: u32,
    /// Number of sub inodes that are directories.
    num_sub_dirs: u32,

    /// ExFAT uses UTF-16 encoding, rust use utf-8 for string processing.
    name: String,

    /// Flag for whether the inode is deleted.
    is_deleted: bool,

    /// The hash of its parent inode.
    parent_hash: usize,

    /// A pointer to exFAT fs.
    fs: Weak<LearnedFS>,

    /// Important: To enlarge the page_cache, we need to update the page_cache size before we update the size of inode, to avoid extra data read.
    /// To shrink the page_cache, we need to update the page_cache size after we update the size of inode, to avoid extra data write.
    page_cache: PageCache,

    model: Model,
    dentry_buffer: Option<HashMap<String, (Ino, InodeType)>>,
    segments: LinkedList<Arc<RwLock<LearnedSegment>>>,
}

impl PageCacheBackend for LearnedInode {
    fn read_page_async(&self, idx: usize, frame: &CachePage) -> Result<BioWaiter> {
        let inner = self.inner.read();
        if inner.size < idx * PAGE_SIZE {
            return_errno_with_message!(Errno::EINVAL, "Invalid read size")
        }
        let sector_id = inner.get_sector_id(idx * PAGE_SIZE / inner.fs().sector_size())?;
        let bio_segment = BioSegment::new_from_segment(
            Segment::from(frame.clone()).into(),
            BioDirection::FromDevice,
        );
        let waiter = inner.fs().block_device().read_blocks_async(
            BlockId::from_offset(sector_id * inner.fs().sector_size()),
            bio_segment,
        )?;
        Ok(waiter)
    }

    fn write_page_async(&self, idx: usize, frame: &CachePage) -> Result<BioWaiter> {
        let inner = self.inner.read();
        let sector_size = inner.fs().sector_size();

        let sector_id = inner.get_sector_id(idx * PAGE_SIZE / inner.fs().sector_size())?;

        // FIXME: We may need to truncate the file if write_page fails.
        // To fix this issue, we need to change the interface of the PageCacheBackend trait.
        let bio_segment = BioSegment::new_from_segment(
            Segment::from(frame.clone()).into(),
            BioDirection::ToDevice,
        );
        let waiter = inner.fs().block_device().write_blocks_async(
            BlockId::from_offset(sector_id * inner.fs().sector_size()),
            bio_segment,
        )?;
        Ok(waiter)
    }

    fn npages(&self) -> usize {
        self.inner.read().size.align_up(PAGE_SIZE) / PAGE_SIZE
    }
}

impl LearnedInodeInner {
    /// The hash_value to index inode. This should be unique in the whole fs.
    /// Currently use inode number as the hash value.
    fn hash_index(&self) -> usize {
        return self.ino as usize;
    }

    fn get_parent_inode(&self) -> Option<Arc<LearnedInode>> {
        //FIXME: What if parent inode is evicted? How can I find it?
        self.fs().find_opened_inode(self.parent_hash)
    }

    /// Get physical sector id from logical sector id for this Inode.
    fn get_sector_id(&self, sector_id: usize) -> Result<usize> {
        let chain_offset = self
            .start_chain
            .walk_to_cluster_at_offset(sector_id * self.fs().sector_size())?;

        let sect_per_cluster = self.fs().super_block().sect_per_cluster as usize;
        let cluster_id = sector_id / sect_per_cluster;
        let cluster = self.get_physical_cluster((sector_id / sect_per_cluster) as ClusterID)?;

        let sec_offset = sector_id % (self.fs().super_block().sect_per_cluster as usize);
        Ok(self.fs().cluster_to_off(cluster) / self.fs().sector_size() + sec_offset)
    }

    /// Get the physical cluster id from the logical cluster id in the inode.
    fn get_physical_cluster(&self, logical: ClusterID) -> Result<ClusterID> {
        let chain = self.start_chain.walk(logical)?;
        Ok(chain.cluster_id())
    }

    /// The number of clusters allocated.
    fn num_clusters(&self) -> u32 {
        self.start_chain.num_clusters()
    }

    fn is_sync(&self) -> bool {
        false
    }

    fn fs(&self) -> Arc<LearnedFS> {
        self.fs.upgrade().unwrap()
    }

    /// Only valid for directory, check if the dir is empty.
    fn is_empty_dir(&self) -> Result<bool> {
        if !self.inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR)
        }
        Ok(self.num_sub_inodes == 0)
    }

    fn make_mode(&self) -> InodeMode {
        self.attr
            .make_mode(self.fs().mount_option(), InodeMode::all())
    }

    fn count_num_sub_inode_and_dir(&self, fs_guard: &MutexGuard<()>) -> Result<(usize, usize)> {
        if !self.start_chain.is_current_cluster_valid() {
            return Ok((0, 0));
        }
        // TODO: may need to init segment header

        let mut sub_inodes = 0;
        let mut sub_dirs = 0;
        for segment in self.segments.iter() {
            let segment_inner = segment.read();
            if segment_inner.total_entry_count > 0 {
                sub_inodes += segment_inner.valid_entry_count;
                sub_dirs += segment_inner
                    .dentries
                    .iter()
                    .filter(|dentry| dentry.attr.contains(DentryAttr::DIRECTORY))
                    .count();
            }
        }
        Ok((sub_inodes, sub_dirs))
    }

    /// Resize current inode to new_size.
    /// The `size_allocated` field in inode can be enlarged, while the `size` field will not.
    fn resize(&mut self, new_size: usize, fs_guard: &MutexGuard<()>) -> Result<()> {
        let fs = self.fs();
        let cluster_size = fs.cluster_size();

        let num_clusters = self.num_clusters();
        let new_num_clusters = (new_size.align_up(cluster_size) / cluster_size) as u32;

        let sync = self.is_sync();

        match new_num_clusters.cmp(&num_clusters) {
            Ordering::Greater => {
                // New clusters should be allocated.
                self.start_chain
                    .extend_clusters(new_num_clusters - num_clusters, sync)?;
            }
            Ordering::Less => {
                // Some exist clusters should be truncated.
                self.start_chain
                    .remove_clusters_from_tail(num_clusters - new_num_clusters, sync)?;
                if new_size < self.size {
                    // Valid data is truncated.
                    self.size = new_size;
                }
            }
            _ => {}
        };
        self.size_allocated = new_size;

        Ok(())
    }

    /// Update inode information back to the disk to sync this inode.
    /// Should lock the file system before calling this function.
    fn write_inode(&self, sync: bool, fs_guard: &MutexGuard<()>) -> Result<()> {
        // Root dir should not be updated.
        if self.ino == LEARNED_ROOT_INO {
            return Ok(());
        }

        // If the inode is deleted, we should not write it back.
        if self.is_deleted {
            return Ok(());
        }

        let raw_inode = LearnedInodeOnDisk {
            valid_size: self.size as u64,
            start_cluster: self.start_chain.cluster_id(),
            reserved: 0,
            create_time: self.ctime.time,
            create_date: self.ctime.date,
            modify_time: self.mtime.time,
            modify_date: self.mtime.date,
            access_time: self.atime.time,
            access_date: self.atime.date,
            create_time_cs: self.ctime.increment_10ms,
            modify_time_cs: self.mtime.increment_10ms,
            create_utc_offset: self.ctime.utc_offset,
            modify_utc_offset: self.mtime.utc_offset,
            access_utc_offset: self.atime.utc_offset,
            flags: self.start_chain.flags().bits(),
        };
        let fs = self.fs();
        fs.write_inode(
            self.ino,
            &raw_inode,
            sync,
        )?;
        Ok(())
    }

    /// Read all sub-inodes from the given position(offset) in this directory.
    /// The number of inodes to read is given by dir_cnt.
    /// Return (the new offset after read, the number of sub-inodes read).
    /// used for readdir and update_subdir_parent_hash
    fn visit_sub_inodes(
        &self,
        offset: usize,
        dir_cnt: usize,
        visitor: &mut dyn DirentVisitor,
        fs_guard: &MutexGuard<()>,
    ) -> Result<(usize, usize)> {
        if !self.inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR)
        }
        if dir_cnt == 0 {
            return Ok((offset, 0));
        }

        let mut rest = dir_cnt;
        let mut iter_start_offset_in_segment = offset;
        let mut dentry_global_offset = 0;
        for segment in self.segments.iter() {
            let segment_inner = segment.read();
            if iter_start_offset_in_segment >= segment_inner.total_entry_count {
                iter_start_offset_in_segment -= segment_inner.total_entry_count;
                dentry_global_offset += segment_inner.total_entry_count;
                continue;
            }
            for i in iter_start_offset_in_segment..segment_inner.total_entry_count {
                let dentry = &segment_inner.dentries[i];
                if dentry.is_deleted() {
                    dentry_global_offset += 1;
                    continue;
                }
                self.visit_sub_inode(dentry.ino, dentry, dentry_global_offset, visitor, fs_guard)?;
                dentry_global_offset += 1;
                rest -= 1;
                if rest == 0 {
                    return Ok((dentry_global_offset, dir_cnt));
                }
            }
            iter_start_offset_in_segment = 0;
        }
        Ok((dentry_global_offset, dir_cnt - rest))
    }

    /// Visit a sub-inode at offset. Return the dentry-set size of the sub-inode.
    /// Dirent visitor will extract information from the inode.
    /// only used in "visit_sub_inodes"
    fn visit_sub_inode(
        &self,
        ino: Ino,
        dentry: &DirEntry,
        offset: usize,
        visitor: &mut dyn DirentVisitor,
        fs_guard: &MutexGuard<()>,
    ) -> Result<()> {
        if !self.inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR)
        }

        if let Some(child_inode) = self.fs().find_opened_inode(ino as usize) {
            // Inode already exists.
            let child_inner = child_inode.inner.read();
            visitor.visit(
                &dentry.name,
                child_inner.ino,
                child_inner.inode_type,
                offset,
            )?;
            
        } else {
            // Otherwise, create a new node and insert it to hash map.
            let fs = self.fs();
            let raw_inode = fs.read_inode(ino)?;
            let inode = LearnedInode::build_from_inode_on_disk(
                fs.clone(),
                &raw_inode,
                ino,
                dentry,
                self.hash_index(),
                fs_guard,
            )?;
            let _ = fs.insert_inode(inode.clone());
            let child_inner = inode.inner.read();
            visitor.visit(
                &dentry.name,
                ino,
                child_inner.inode_type,
                offset,
            )?;
        }
        return Ok(());
    }

    fn sync_metadata(&self, fs_guard: &MutexGuard<()>) -> Result<()> {
        self.fs().bitmap().lock().sync()?;
        self.write_inode(true, fs_guard)?;
        Ok(())
    }

    fn sync_data(&self, fs_guard: &MutexGuard<()>) -> Result<()> {
        self.page_cache.evict_range(0..self.size)?;
        Ok(())
    }

    fn sync_all(&self, fs_guard: &MutexGuard<()>) -> Result<()> {
        self.sync_metadata(fs_guard)?;
        self.sync_data(fs_guard)?;
        Ok(())
    }

    /// Update the metadata for current directory after a delete.
    /// Set is_dir if the deleted file is a directory.
    fn update_metadata_for_delete(&mut self, is_dir: bool) {
        self.num_sub_inodes -= 1;
        if is_dir {
            self.num_sub_dirs -= 1;
        }
    }

    fn update_atime(&mut self) -> Result<()> {
        self.atime = DosTimestamp::now()?;
        Ok(())
    }

    fn update_atime_and_mtime(&mut self) -> Result<()> {
        let now = DosTimestamp::now()?;
        self.atime = now;
        self.mtime = now;
        Ok(())
    }

    fn merge_buffer(&mut self) -> Result<()> {
        if !self.inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR);
        }

        if self.dentry_buffer.is_none() {
            return Ok(());
        }

        let mut new_entries = self.dentry_buffer
            .take()
            .unwrap()
            .drain()
            .map(|(name, (ino, _type))| DirEntry::new(name, ino, _type))
            .collect::<Vec<_>>();
        new_entries.sort_unstable();

        let mut segment_cursor = self.segments.cursor_front_mut();
        let mut process_idx = 0;
        while let Some(_) = segment_cursor.current() {
            let (first_entry, last_entry) = {
                let segment = segment_cursor.current().unwrap();
                (
                    segment.read().dentries.first().unwrap().clone(),
                    segment.read().dentries.last().unwrap().clone(),
                )
            };
            // number of entries before the current segment
            let mut before_cnt = new_entries[process_idx..].binary_search_by(
                |entry| entry.cmp(&first_entry)
            ).unwrap_or_else(|x| x);
            
            while before_cnt >= MIN_SEGMENT_SIZE {
                let new_segment = LearnedSegment::new(&new_entries[process_idx..process_idx + MIN_SEGMENT_SIZE], self.fs.clone());
                self.model.train(new_segment.clone())?;
                segment_cursor.insert_before(new_segment);
                process_idx += MIN_SEGMENT_SIZE;
                before_cnt -= MIN_SEGMENT_SIZE;
            }

            // number of entries falls in the current segment (before_cnt is included)
            let in_cnt = new_entries[process_idx..].binary_search_by(
                |entry| entry.cmp(&last_entry)
            ).unwrap_or_else(|x| x);
            
            // new entries falls in the middle of current segment and next segment
            let after_cnt = if let Some(next_seg) = segment_cursor.peek_next() {
                let binding = next_seg.read();
                let next_first_entry = binding.dentries.first().unwrap();
                let after = new_entries[process_idx..].binary_search_by(
                    |entry| entry.cmp(&next_first_entry)
                ).unwrap_or_else(|x| x) - in_cnt;
                after % MIN_SEGMENT_SIZE
            } else {
                let after = new_entries.len() - process_idx - in_cnt;
                after % MIN_SEGMENT_SIZE
            };

            if in_cnt + after_cnt > 0 {
                // merge in_cnt+after_cnt entries into current segment
                // Uninstall the current segment from cursor
                let cur_segment = segment_cursor.remove_current().unwrap();
                self.model.remove(cur_segment.clone())?;
                // Merge current segment with new entries
                if let Ok(cur_segment) = Arc::try_unwrap(cur_segment) {
                    let new_segments = cur_segment.read().merge(&new_entries[process_idx..process_idx + in_cnt + after_cnt]);
                    for new_segment in new_segments {
                        self.model.train(new_segment.clone())?;
                        segment_cursor.insert_before(new_segment);
                    }
                    process_idx += in_cnt + after_cnt;
                } else {
                    panic!("Failed to unwrap current segment, this should not happen");
                }
            } else {
                segment_cursor.move_next();
            }
        }

        while process_idx < new_entries.len() {
            let new_segment = LearnedSegment::new(&new_entries[process_idx..process_idx + MIN_SEGMENT_SIZE], self.fs.clone());
            self.model.train(new_segment.clone())?;
            self.segments.push_back(new_segment);
            process_idx += MIN_SEGMENT_SIZE;
        }
        
        Ok(())
    }


    pub(super) fn lookup_entry(&self, name: &str, fs_guard: &MutexGuard<()>) -> Result<Arc<LearnedInode>> {
        let fs = self.fs();
        if !self.inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR);
        }

        if let Some(dentry_buffer) = &self.dentry_buffer {
            if let Some((ino, _type)) = dentry_buffer.get(name) {
                return Ok(fs.find_opened_inode(*ino as usize).unwrap());
            }
        }

        let predicted = self.model.predict_position(name)?;
        let binding = predicted.segment.read();
        let dentry = binding.search(predicted.offset, name)?;
        if dentry.is_deleted() {
            return_errno!(Errno::ENOENT);
        }
        return Ok(fs.find_opened_inode(dentry.ino() as usize).unwrap());
    }
}

impl LearnedInode {
    // TODO: Should be called when inode is evicted from fs.
    pub(super) fn reclaim_space(&self) -> Result<()> {
        let inner = self.inner.write();
        let fs = inner.fs();
        let fs_guard = fs.lock();
        self.inner.write().resize(0, &fs_guard)?;
        self.inner.read().page_cache.resize(0)?;
        Ok(())
    }

    pub(super) fn hash_index(&self) -> usize {
        self.inner.read().hash_index()
    }

    pub(super) fn is_deleted(&self) -> bool {
        // TODO: used DELETE flag from dentryattr.
        self.inner.read().is_deleted
    }

    pub(super) fn build_root_inode(
        fs_weak: Weak<LearnedFS>,
        root_chain: ExfatChain,
    ) -> Result<Arc<LearnedInode>> {
        let sb = fs_weak.upgrade().unwrap().super_block();

        let dentry_set_size = 0;

        let attr = DentryAttr::DIRECTORY;

        let inode_type = InodeType::Dir;

        let ctime = DosTimestamp::now()?;

        let size = root_chain.num_clusters() as usize * sb.cluster_size as usize;

        let name = String::new();

        let inode = Arc::new_cyclic(|weak_self| LearnedInode {
            inner: RwMutex::new(LearnedInodeInner {
                ino: LEARNED_ROOT_INO,
                dentry_set_position: ExfatChainPosition::default(),
                dentry_set_size: 0,
                dentry_entry: 0,
                inode_type,
                attr,
                start_chain: root_chain,
                size,
                size_allocated: size,
                atime: ctime,
                mtime: ctime,
                ctime,
                num_sub_inodes: 0,
                num_sub_dirs: 0,
                name,
                is_deleted: false,
                parent_hash: 0,
                fs: fs_weak,
                page_cache: PageCache::with_capacity(size, weak_self.clone() as _).unwrap(),
                model: Model::new(),
                dentry_buffer: None,
                segments: LinkedList::new(),
            }),
            extension: Extension::new(),
        });

        let inner = inode.inner.upread();
        let fs = inner.fs();
        let fs_guard = fs.lock();

        let num_sub_inode_dir: (usize, usize) = inner.count_num_sub_inode_and_dir(&fs_guard)?;

        let mut inode_inner = inner.upgrade();

        inode_inner.num_sub_inodes = num_sub_inode_dir.0 as u32;
        inode_inner.num_sub_dirs = num_sub_inode_dir.1 as u32;

        Ok(inode.clone())
    }

    fn build_from_inode_on_disk(
        fs: Arc<LearnedFS>,
        inode_on_disk: &LearnedInodeOnDisk,
        ino: Ino,
        dentry: &DirEntry,
        parent_hash: usize,
        fs_guard: &MutexGuard<()>,
    ) -> Result<Arc<Self>> {
        let fs_weak = Arc::downgrade(&fs);
        let attr = dentry.attr;
        let inode_type = attr.make_type();

        let ctime = DosTimestamp::new(
            inode_on_disk.create_time,
            inode_on_disk.create_date,
            inode_on_disk.create_time_cs,
            inode_on_disk.create_utc_offset,
        )?;
        let mtime = DosTimestamp::new(
            inode_on_disk.modify_time,
            inode_on_disk.modify_date,
            inode_on_disk.modify_time_cs,
            inode_on_disk.modify_utc_offset,
        )?;
        let atime = DosTimestamp::new(
            inode_on_disk.access_time,
            inode_on_disk.access_date,
            0,
            inode_on_disk.access_utc_offset,
        )?;

        let size = inode_on_disk.valid_size as usize;
        let chain_flag = FatChainFlags::from_bits_truncate(inode_on_disk.flags);
        let start_cluster = inode_on_disk.start_cluster;
        let num_clusters = size.align_up(fs.cluster_size()) / fs.cluster_size();
        let start_chain = ExfatChain::new(
            fs_weak.clone(),
            start_cluster,
            Some(num_clusters as u32),
            chain_flag,
        )?;
        let name = dentry.name.clone();
        let inode = Arc::new_cyclic(|weak_self| LearnedInode {
            inner: RwMutex::new(LearnedInodeInner {
                ino,
                dentry_set_position: ExfatChainPosition::default(),
                dentry_set_size: 0,
                dentry_entry: 0,
                inode_type,
                attr,
                start_chain,
                size,
                size_allocated: size,
                atime,
                mtime,
                ctime,
                num_sub_inodes: 0,
                num_sub_dirs: 0,
                name,
                is_deleted: false,
                parent_hash,
                fs: fs_weak,
                page_cache: PageCache::with_capacity(size, weak_self.clone() as _).unwrap(),
                model: Model::new(),
                dentry_buffer: None,
                segments: LinkedList::new(),
            }),
            extension: Extension::new(),
        });

        if matches!(inode_type, InodeType::Dir) {
            let inner = inode.inner.upread();
            let num_sub_inode_dir: (usize, usize) = inner.count_num_sub_inode_and_dir(fs_guard)?;

            let mut inode_inner = inner.upgrade();

            inode_inner.num_sub_inodes = num_sub_inode_dir.0 as u32;
            inode_inner.num_sub_dirs = num_sub_inode_dir.1 as u32;
        }

        Ok(inode)
    }

    /// Add new dentries. Create a new file or folder.
    fn add_entry(
        &self,
        name: &str,
        inode_type: InodeType,
        mode: InodeMode,
        fs_guard: &MutexGuard<()>,
    ) -> Result<Arc<LearnedInode>> {
        if name.len() > MAX_NAME_LENGTH {
            return_errno!(Errno::ENAMETOOLONG)
        }

        let fs = self.inner.read().fs();

        let raw_inode = LearnedInodeOnDisk::new(fs.clone())?;
        let temp_dentry = DirEntry::new(
            name.to_string(),
            0, // inode number will be assigned later
            inode_type,
        );
        let ino = fs.alloc_inode_number();
        let inode = Self::build_from_inode_on_disk(
            fs.clone(),
            &raw_inode,
            ino,
            &temp_dentry,
            self.hash_index(),
            fs_guard,
        )?;

        let mut inner = self.inner.write();
        if inner.dentry_buffer.is_none() {
            inner.dentry_buffer = Some(HashMap::new());
        }

        inner.num_sub_inodes += 1;
        if inode_type.is_directory() {
            inner.num_sub_dirs += 1;
        }

        let dentry_buffer = inner.dentry_buffer.as_mut().unwrap();
        let ino = inode.ino();
        dentry_buffer.insert(name.to_string(), (ino, inode_type));


        if dentry_buffer.len() >= MAX_BUFFER_SIZE {
            inner.merge_buffer()?;
        }

        Ok(inode)
    }

    /// Copy metadata from the given inode.
    /// There will be no deadlock since this function is only used in rename and the arg "inode".
    /// is a temporary inode which is only accessible to current thread.
    /// used in rename only
    fn copy_metadata_from(&self, inode: Arc<LearnedInode>) {
        let mut self_inner = self.inner.write();
        let other_inner = inode.inner.read();

        self_inner.dentry_set_position = other_inner.dentry_set_position.clone();
        self_inner.dentry_set_size = other_inner.dentry_set_size;
        self_inner.dentry_entry = other_inner.dentry_entry;
        self_inner.atime = other_inner.atime;
        self_inner.ctime = other_inner.ctime;
        self_inner.mtime = other_inner.mtime;
        self_inner.name = other_inner.name.clone();
        self_inner.is_deleted = other_inner.is_deleted;
        self_inner.parent_hash = other_inner.parent_hash;
    }

    // used in rename only
    fn update_subdir_parent_hash(&self, fs_guard: &MutexGuard<()>) -> Result<()> {
        let inner = self.inner.read();
        if !inner.inode_type.is_directory() {
            return Ok(());
        }
        let new_parent_hash = self.hash_index();
        let sub_dir = inner.num_sub_inodes;
        let mut child_inodes = InoVisitor {
            inodes: vec![],
        };
        inner.visit_sub_inodes(0, sub_dir as usize, &mut child_inodes, fs_guard)?;

        let start_chain = inner.start_chain.clone();
        for ino in child_inodes.inodes {
            let child_inode = inner.fs().find_opened_inode(ino as usize).unwrap();
            child_inode.inner.write().parent_hash = new_parent_hash;
        }
        Ok(())
    }

    /// Unlink a file or remove a directory.
    /// Need to delete dentry set and inode.
    /// Delete the file contents if delete_content is set.
    fn delete_entry(
        &self,
        inode: Arc<LearnedInode>,
        delete_contents: bool,
        fs_guard: &MutexGuard<()>,
    ) -> Result<()> {
        // Delete directory contents directly.
        let is_dir = inode.inner.read().inode_type.is_directory();
        if delete_contents {
            if is_dir {
                inode.inner.write().resize(0, fs_guard)?;
                inode.inner.read().page_cache.resize(0)?;
            }
            // Set the delete flag.
            inode.inner.write().is_deleted = true;
        }
        // Remove the inode.
        self.inner.read().fs().remove_inode(inode.hash_index());
        // Try to remove the dentry from the buffer.
        let name = inode.inner.read().name.to_string();
        if let Some(dentry_buffer) = &mut self.inner.write().dentry_buffer {
            if let Some(ino) = dentry_buffer.remove(&name) {
                self.inner.write().update_metadata_for_delete(is_dir);
                return Ok(());
            }
        }
        // If not found in dentry buffer, we need to delete the dentry from the segment.
        let predicted = self.inner.read().model.predict_position(&name)?;
        let mut binding = predicted.segment.write();
        let dentry = binding.search_mut(predicted.offset, &name)?;
        dentry.mark_deleted();
        self.inner.write().update_metadata_for_delete(is_dir);
        Ok(())
    }
}

struct EmptyVisitor;
impl DirentVisitor for EmptyVisitor {
    fn visit(&mut self, name: &str, ino: u64, type_: InodeType, offset: usize) -> Result<()> {
        Ok(())
    }
}
struct InoVisitor {
    inodes: Vec<Ino>,
}
impl DirentVisitor for InoVisitor {
    fn visit(&mut self, _name: &str, ino: u64, _type_: InodeType, _offset: usize) -> Result<()> {
        self.inodes.push(ino);
        Ok(())
    }
}
fn is_block_aligned(off: usize) -> bool {
    off % PAGE_SIZE == 0
}

fn check_corner_cases_for_rename(
    old_inode: &Arc<LearnedInode>,
    exist_inode: &Arc<LearnedInode>,
) -> Result<()> {
    // Check for two corner cases here.
    let old_inode_is_dir = old_inode.inner.read().inode_type.is_directory();
    // If old_inode represents a directory, the exist 'new_name' must represents a empty directory.
    if old_inode_is_dir && !exist_inode.inner.read().is_empty_dir()? {
        return_errno!(Errno::ENOTEMPTY)
    }
    // If old_inode represents a file, the exist 'new_name' must also represents a file.
    if !old_inode_is_dir && exist_inode.inner.read().inode_type.is_directory() {
        return_errno!(Errno::EISDIR)
    }
    Ok(())
}

impl Inode for LearnedInode {
    fn ino(&self) -> u64 {
        self.inner.read().ino
    }

    fn size(&self) -> usize {
        self.inner.read().size
    }

    fn resize(&self, new_size: usize) -> Result<()> {
        let inner = self.inner.upread();

        if inner.inode_type.is_directory() {
            return_errno!(Errno::EISDIR)
        }

        let file_size = inner.size;
        let fs = inner.fs();
        let fs_guard = fs.lock();

        inner.upgrade().resize(new_size, &fs_guard)?;

        // Update the size of page cache.
        let inner = self.inner.read();

        // We will delay updating the page_cache size when enlarging an inode until the real write.
        if new_size < file_size {
            self.inner.read().page_cache.resize(new_size)?;
        }

        // Sync this inode since size has changed.
        if inner.is_sync() {
            inner.sync_metadata(&fs_guard)?;
        }

        Ok(())
    }

    fn metadata(&self) -> crate::fs::utils::Metadata {
        let inner = self.inner.read();

        let blk_size = inner.fs().super_block().sector_size as usize;

        let nlinks = if inner.inode_type.is_directory() {
            (inner.num_sub_dirs + 2) as usize
        } else {
            1
        };

        Metadata {
            dev: 0,
            ino: inner.ino,
            size: inner.size,
            blk_size,
            blocks: inner.size.div_ceil(blk_size),
            atime: inner.atime.as_duration().unwrap_or_default(),
            mtime: inner.mtime.as_duration().unwrap_or_default(),
            ctime: inner.ctime.as_duration().unwrap_or_default(),
            type_: inner.inode_type,
            mode: inner.make_mode(),
            nlinks,
            uid: Uid::new(inner.fs().mount_option().fs_uid as u32),
            gid: Gid::new(inner.fs().mount_option().fs_gid as u32),
            //real device
            rdev: 0,
        }
    }

    fn type_(&self) -> InodeType {
        self.inner.read().inode_type
    }

    fn mode(&self) -> Result<InodeMode> {
        Ok(self.inner.read().make_mode())
    }

    fn set_mode(&self, mode: InodeMode) -> Result<()> {
        //Pass through
        Ok(())
    }

    fn atime(&self) -> Duration {
        self.inner.read().atime.as_duration().unwrap_or_default()
    }

    fn set_atime(&self, time: Duration) {
        self.inner.write().atime = DosTimestamp::from_duration(time).unwrap_or_default();
    }

    fn mtime(&self) -> Duration {
        self.inner.read().mtime.as_duration().unwrap_or_default()
    }

    fn set_mtime(&self, time: Duration) {
        self.inner.write().mtime = DosTimestamp::from_duration(time).unwrap_or_default();
    }

    fn ctime(&self) -> Duration {
        self.inner.read().ctime.as_duration().unwrap_or_default()
    }

    fn set_ctime(&self, time: Duration) {
        self.inner.write().ctime = DosTimestamp::from_duration(time).unwrap_or_default();
    }

    fn owner(&self) -> Result<Uid> {
        Ok(Uid::new(
            self.inner.read().fs().mount_option().fs_uid as u32,
        ))
    }

    fn set_owner(&self, uid: Uid) -> Result<()> {
        // Pass through.
        Ok(())
    }

    fn group(&self) -> Result<Gid> {
        Ok(Gid::new(
            self.inner.read().fs().mount_option().fs_gid as u32,
        ))
    }

    fn set_group(&self, gid: Gid) -> Result<()> {
        // Pass through.
        Ok(())
    }

    fn fs(&self) -> alloc::sync::Arc<dyn crate::fs::utils::FileSystem> {
        self.inner.read().fs()
    }

    fn page_cache(&self) -> Option<Vmo<Full>> {
        Some(self.inner.read().page_cache.pages().dup())
    }

    fn read_at(&self, offset: usize, writer: &mut VmWriter) -> Result<usize> {
        let inner = self.inner.upread();
        if inner.inode_type.is_directory() {
            return_errno!(Errno::EISDIR)
        }
        let (read_off, read_len) = {
            let file_size = inner.size;
            let start = file_size.min(offset);
            let end = file_size.min(offset + writer.avail());
            (start, end - start)
        };
        inner.page_cache.pages().read(read_off, writer)?;

        inner.upgrade().update_atime()?;
        Ok(read_len)
    }

    // The offset and the length of buffer must be multiples of the block size.
    fn read_direct_at(&self, offset: usize, writer: &mut VmWriter) -> Result<usize> {
        let inner = self.inner.upread();
        if inner.inode_type.is_directory() {
            return_errno!(Errno::EISDIR)
        }
        if !is_block_aligned(offset) || !is_block_aligned(writer.avail()) {
            return_errno_with_message!(Errno::EINVAL, "not block-aligned");
        }

        let sector_size = inner.fs().sector_size();

        let (read_off, read_len) = {
            let file_size = inner.size;
            let start = file_size.min(offset).align_down(sector_size);
            let end = file_size
                .min(offset + writer.avail())
                .align_down(sector_size);
            (start, end - start)
        };

        inner
            .page_cache
            .discard_range(read_off..read_off + read_len);

        let mut buf_offset = 0;
        let bio_segment = BioSegment::alloc(1, BioDirection::FromDevice);

        let start_pos = inner.start_chain.walk_to_cluster_at_offset(read_off)?;
        let cluster_size = inner.fs().cluster_size();
        let mut cur_cluster = start_pos.0.clone();
        let mut cur_offset = start_pos.1;
        for _ in Bid::from_offset(read_off)..Bid::from_offset(read_off + read_len) {
            let physical_bid =
                Bid::from_offset(cur_cluster.cluster_id() as usize * cluster_size + cur_offset);
            inner
                .fs()
                .block_device()
                .read_blocks(physical_bid, bio_segment.clone())?;
            bio_segment.reader().unwrap().read_fallible(writer)?;
            buf_offset += BLOCK_SIZE;

            cur_offset += BLOCK_SIZE;
            if cur_offset >= cluster_size {
                cur_cluster = cur_cluster.walk(1)?;
                cur_offset %= BLOCK_SIZE;
            }
        }

        inner.upgrade().update_atime()?;
        Ok(read_len)
    }

    fn write_at(&self, offset: usize, reader: &mut VmReader) -> Result<usize> {
        let write_len = reader.remain();
        // We need to obtain the fs lock to resize the file.
        let new_size = {
            let mut inner = self.inner.write();
            if inner.inode_type.is_directory() {
                return_errno!(Errno::EISDIR)
            }

            let file_size = inner.size;
            let file_allocated_size = inner.size_allocated;
            let new_size = offset + write_len;
            let fs = inner.fs();
            let fs_guard = fs.lock();
            if new_size > file_size {
                if new_size > file_allocated_size {
                    inner.resize(new_size, &fs_guard)?;
                }
                inner.page_cache.resize(new_size)?;
            }
            new_size.max(file_size)
        };

        // Locks released here, so that file write can be parallelized.
        let inner = self.inner.upread();
        inner.page_cache.pages().write(offset, reader)?;

        // Update timestamps and size.
        {
            let mut inner = inner.upgrade();

            inner.update_atime_and_mtime()?;
            inner.size = new_size;
        }

        let inner = self.inner.read();

        // Write data back.
        if inner.is_sync() {
            let fs = inner.fs();
            let fs_guard = fs.lock();
            inner.sync_all(&fs_guard)?;
        }

        Ok(write_len)
    }

    fn write_direct_at(&self, offset: usize, reader: &mut VmReader) -> Result<usize> {
        let write_len = reader.remain();
        let inner = self.inner.upread();
        if inner.inode_type.is_directory() {
            return_errno!(Errno::EISDIR)
        }
        if !is_block_aligned(offset) || !is_block_aligned(write_len) {
            return_errno_with_message!(Errno::EINVAL, "not block-aligned");
        }

        let file_size = inner.size;
        let file_allocated_size = inner.size_allocated;
        let end_offset = offset + write_len;

        let start = offset.min(file_size);
        let end = end_offset.min(file_size);
        inner.page_cache.discard_range(start..end);

        let new_size = {
            let mut inner = inner.upgrade();
            if end_offset > file_size {
                let fs = inner.fs();
                let fs_guard = fs.lock();
                if end_offset > file_allocated_size {
                    inner.resize(end_offset, &fs_guard)?;
                }
                inner.page_cache.resize(end_offset)?;
            }
            file_size.max(end_offset)
        };

        let inner = self.inner.upread();

        let bio_segment = BioSegment::alloc(1, BioDirection::ToDevice);
        let start_pos = inner.start_chain.walk_to_cluster_at_offset(offset)?;
        let cluster_size = inner.fs().cluster_size();
        let mut cur_cluster = start_pos.0.clone();
        let mut cur_offset = start_pos.1;
        for _ in Bid::from_offset(offset)..Bid::from_offset(end_offset) {
            bio_segment.writer().unwrap().write_fallible(reader)?;
            let physical_bid =
                Bid::from_offset(cur_cluster.cluster_id() as usize * cluster_size + cur_offset);
            let fs = inner.fs();
            fs.block_device()
                .write_blocks(physical_bid, bio_segment.clone())?;

            cur_offset += BLOCK_SIZE;
            if cur_offset >= cluster_size {
                cur_cluster = cur_cluster.walk(1)?;
                cur_offset %= BLOCK_SIZE;
            }
        }

        {
            let mut inner = inner.upgrade();
            inner.update_atime_and_mtime()?;
            inner.size = new_size;
        }

        let inner = self.inner.read();
        // Sync this inode since size has changed.
        if inner.is_sync() {
            let fs = inner.fs();
            let fs_guard = fs.lock();
            inner.sync_metadata(&fs_guard)?;
        }

        Ok(write_len)
    }

    fn create(&self, name: &str, type_: InodeType, mode: InodeMode) -> Result<Arc<dyn Inode>> {
        let fs = self.inner.read().fs();
        let fs_guard = fs.lock();
        {
            let inner = self.inner.read();
            if !inner.inode_type.is_directory() {
                return_errno!(Errno::ENOTDIR)
            }
            if name.len() > MAX_NAME_LENGTH {
                return_errno!(Errno::ENAMETOOLONG)
            }

            if inner.lookup_entry(name, &fs_guard).is_ok() {
                return_errno!(Errno::EEXIST)
            }
        }

        let result = self.add_entry(name, type_, mode, &fs_guard)?;
        let _ = fs.insert_inode(result.clone());

        self.inner.write().update_atime_and_mtime()?;

        let inner = self.inner.read();

        if inner.is_sync() {
            inner.sync_all(&fs_guard)?;
        }

        Ok(result)
    }

    fn mknod(&self, name: &str, mode: InodeMode, type_: MknodType) -> Result<Arc<dyn Inode>> {
        return_errno_with_message!(Errno::EINVAL, "unsupported operation")
    }

    fn readdir_at(&self, dir_cnt: usize, visitor: &mut dyn DirentVisitor) -> Result<usize> {
        let inner = self.inner.upread();

        if dir_cnt >= (inner.num_sub_inodes + 2) as usize {
            return Ok(0);
        }

        let mut empty_visitor = EmptyVisitor;

        let dir_read = {
            let fs = inner.fs();
            let fs_guard = fs.lock();

            let mut dir_read = 0usize;

            if dir_cnt == 0
                && visitor
                    .visit(".", inner.ino, inner.inode_type, 0xFFFFFFFFFFFFFFFEusize)
                    .is_ok()
            {
                dir_read += 1;
            }

            if dir_cnt <= 1 {
                let parent_inode = inner.get_parent_inode().unwrap();
                let parent_inner = parent_inode.inner.read();
                let ino = parent_inner.ino;
                let type_ = parent_inner.inode_type;
                if visitor
                    .visit("..", ino, type_, 0xFFFFFFFFFFFFFFFFusize)
                    .is_ok()
                {
                    dir_read += 1;
                }
            }

            // Skip . and ..
            let dir_to_skip = if dir_cnt >= 2 { dir_cnt - 2 } else { 0 };

            // Skip previous directories.
            let (off, _) = inner.visit_sub_inodes(0, dir_to_skip, &mut empty_visitor, &fs_guard)?;
            let (_, read) = inner.visit_sub_inodes(
                off,
                inner.num_sub_inodes as usize - dir_to_skip,
                visitor,
                &fs_guard,
            )?;
            dir_read += read;
            dir_read
        };

        inner.upgrade().update_atime()?;

        Ok(dir_read)
    }

    fn link(&self, old: &Arc<dyn Inode>, name: &str) -> Result<()> {
        return_errno_with_message!(Errno::EINVAL, "unsupported operation")
    }

    fn unlink(&self, name: &str) -> Result<()> {
        if !self.inner.read().inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR)
        }
        if name.len() > MAX_NAME_LENGTH {
            return_errno!(Errno::ENAMETOOLONG)
        }
        if is_dot_or_dotdot(name) {
            return_errno!(Errno::EISDIR)
        }

        let fs = self.inner.read().fs();
        let fs_guard = fs.lock();

        let inode = self.inner.read().lookup_entry(name, &fs_guard)?;

        // FIXME: we need to step by following line to avoid deadlock.
        if inode.type_() != InodeType::File {
            return_errno!(Errno::EISDIR)
        }
        self.delete_entry(inode, true, &fs_guard)?;
        self.inner.write().update_atime_and_mtime()?;

        let inner = self.inner.read();
        if inner.is_sync() {
            inner.sync_all(&fs_guard)?;
        }

        Ok(())
    }

    fn rmdir(&self, name: &str) -> Result<()> {
        if !self.inner.read().inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR)
        }
        if is_dot(name) {
            return_errno_with_message!(Errno::EINVAL, "rmdir on .")
        }
        if is_dotdot(name) {
            return_errno_with_message!(Errno::ENOTEMPTY, "rmdir on ..")
        }
        if name.len() > MAX_NAME_LENGTH {
            return_errno!(Errno::ENAMETOOLONG)
        }

        let fs = self.inner.read().fs();
        let fs_guard = fs.lock();

        let inode = self.inner.read().lookup_entry(name, &fs_guard)?;

        if inode.inner.read().inode_type != InodeType::Dir {
            return_errno!(Errno::ENOTDIR)
        } else if !inode.inner.read().is_empty_dir()? {
            // Check if directory to be deleted is empty.
            return_errno!(Errno::ENOTEMPTY)
        }
        self.delete_entry(inode, true, &fs_guard)?;
        self.inner.write().update_atime_and_mtime()?;

        let inner = self.inner.read();
        // Sync this inode since size has changed.
        if inner.is_sync() {
            inner.sync_all(&fs_guard)?;
        }

        Ok(())
    }

    fn lookup(&self, name: &str) -> Result<Arc<dyn Inode>> {
        // FIXME: Readdir should be immutable instead of mutable, but there will be no performance issues due to the global fs lock.
        let inner = self.inner.upread();
        if !inner.inode_type.is_directory() {
            return_errno!(Errno::ENOTDIR)
        }

        if name.len() > MAX_NAME_LENGTH {
            return_errno!(Errno::ENAMETOOLONG)
        }

        let inode = {
            let fs = inner.fs();
            let fs_guard = fs.lock();
            inner.lookup_entry(name, &fs_guard)?
        };

        inner.upgrade().update_atime()?;

        Ok(inode)
    }

    fn rename(&self, old_name: &str, target: &Arc<dyn Inode>, new_name: &str) -> Result<()> {
        if is_dot_or_dotdot(old_name) || is_dot_or_dotdot(new_name) {
            return_errno!(Errno::EISDIR);
        }
        if old_name.len() > MAX_NAME_LENGTH || new_name.len() > MAX_NAME_LENGTH {
            return_errno!(Errno::ENAMETOOLONG)
        }
        let Some(target_) = target.downcast_ref::<LearnedInode>() else {
            return_errno_with_message!(Errno::EINVAL, "not an learned inode")
        };
        if !self.inner.read().inode_type.is_directory()
            || !target_.inner.read().inode_type.is_directory()
        {
            return_errno!(Errno::ENOTDIR)
        }

        let fs = self.inner.read().fs();
        let fs_guard = fs.lock();
        // Rename something to itself, return success directly.
        if self.inner.read().ino == target_.inner.read().ino && old_name.eq(new_name) {
            return Ok(());
        }

        // Read 'old_name' file or dir and its dentries.
        let old_inode = self
            .inner
            .read()
            .lookup_entry(old_name, &fs_guard)?;
        // FIXME: Users may be confused, since inode with the same upper case name will be removed.
        let lookup_exist_result = target_
            .inner
            .read()
            .lookup_entry(new_name, &fs_guard);
        // Check for the corner cases.
        if let Ok(ref exist_inode) = lookup_exist_result {
            check_corner_cases_for_rename(&old_inode, exist_inode)?;
        }

        // All checks are done here. This is a valid rename and it needs to modify the metadata.
        self.delete_entry(old_inode.clone(), false, &fs_guard)?;
        // Create the new dentries.
        let new_inode =
            target_.add_entry(new_name, old_inode.type_(), old_inode.mode()?, &fs_guard)?;
        // Update metadata.
        old_inode.copy_metadata_from(new_inode);
        // Update its children's parent_hash.
        old_inode.update_subdir_parent_hash(&fs_guard)?;
        // Insert back.
        let _ = fs.insert_inode(old_inode.clone());
        // Remove the exist 'new_name' file.
        if let Ok(exist_inode) = lookup_exist_result {
            target_.delete_entry(exist_inode, true, &fs_guard)?;
        }
        // Update the times.
        self.inner.write().update_atime_and_mtime()?;
        target_.inner.write().update_atime_and_mtime()?;
        // Sync
        if self.inner.read().is_sync() || target_.inner.read().is_sync() {
            // TODO: what if fs crashed between syncing?
            old_inode.inner.read().sync_all(&fs_guard)?;
            target_.inner.read().sync_all(&fs_guard)?;
            self.inner.read().sync_all(&fs_guard)?;
        }
        Ok(())
    }

    fn read_link(&self) -> Result<String> {
        return_errno_with_message!(Errno::EINVAL, "unsupported operation")
    }

    fn write_link(&self, target: &str) -> Result<()> {
        return_errno_with_message!(Errno::EINVAL, "unsupported operation")
    }

    fn ioctl(&self, cmd: IoctlCmd, arg: usize) -> Result<i32> {
        return_errno_with_message!(Errno::EINVAL, "unsupported operation")
    }

    fn sync_all(&self) -> Result<()> {
        let inner = self.inner.read();
        let fs = inner.fs();
        let fs_guard = fs.lock();
        inner.sync_all(&fs_guard)?;

        fs.block_device().sync()?;

        Ok(())
    }

    fn sync_data(&self) -> Result<()> {
        let inner = self.inner.read();
        let fs = inner.fs();
        let fs_guard = fs.lock();
        inner.sync_data(&fs_guard)?;

        fs.block_device().sync()?;

        Ok(())
    }

    fn poll(&self, mask: IoEvents, _poller: Option<&mut PollHandle>) -> IoEvents {
        let events = IoEvents::IN | IoEvents::OUT;
        events & mask
    }

    fn is_dentry_cacheable(&self) -> bool {
        true
    }

    fn extension(&self) -> Option<&Extension> {
        Some(&self.extension)
    }
}
