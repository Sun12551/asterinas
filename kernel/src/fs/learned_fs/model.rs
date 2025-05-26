use crate::prelude::*;
use xarray::XArray;
use core::fmt;
use super::{
    segment::LearnedSegment,
    constants::{MAX_MODEL_ERROR, FIXED_POINT_SHIFT},
};
pub(super) type SegmentOffset = usize;
type Slope = f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ModelKey(u64);

impl ModelKey {
    pub(super) fn from_name(name: &str) -> Self {
        let mut val = 0;
        for byte in name.as_bytes() {
            val = (val << 8) | (*byte as u64);
        }
        ModelKey(val)
    }
}

pub(super) struct ModelPosition {
    pub(super) segment: Arc<RwLock<LearnedSegment>>,
    pub(super) offset: SegmentOffset,
    pub(super) slope: Slope,
}

pub(super) struct Model(XArray<Arc<ModelPosition>>);

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model {{ .. }}")
    }
}

impl Model {
    pub(super) fn new() -> Self {
        Self(XArray::new())
    }

    fn calculate_slope(key1: &ModelKey, key2: &ModelKey, ofs1: SegmentOffset, ofs2: SegmentOffset) -> Slope {
        let diff_pos = if ofs2 > ofs1 {
            ((ofs2 - ofs1) << FIXED_POINT_SHIFT) as f64
        } else {
            -(((ofs1 - ofs2) << FIXED_POINT_SHIFT) as f64)
        };
        let diff_key = key2.0 as Slope - key1.0 as Slope;
        diff_pos / diff_key
    }

    fn linear_interpolation(cur_key: &ModelKey, right_key: &ModelKey, right_key_ofs: SegmentOffset, slope: &Slope) -> SegmentOffset {
        let diff_key = right_key.0 as Slope - cur_key.0 as Slope;
        let diff_pos = ((diff_key * slope) as SegmentOffset) >> FIXED_POINT_SHIFT;
        return right_key_ofs - diff_pos;
    }

    pub(super) fn train(&mut self, segment: Arc<RwLock<LearnedSegment>>) -> Result<()> {
        let make_model_position = |offset: SegmentOffset, slope: Slope| {
            Arc::new(ModelPosition {
                segment: segment.clone(),
                offset,
                slope,
            })
        };
        let add_spline_key = |key: &ModelKey, pos: Arc<ModelPosition>| {
            let mut locked = self.0.lock();
            locked.store(key.0, pos);
        };
        let mut last_spline_key = ModelKey(0);
        let mut last_spline_offset = 0;
        let mut last_key = ModelKey(0);
        let mut last_offset = 0;
        let mut offset: SegmentOffset = 0;
        let mut slope_upper_bound = Slope::MAX;
        let mut slope_lower_bound = Slope::MIN;
        for entry in segment.read().dentries.iter() {
            let key = ModelKey::from_name(&entry.name);
            if offset == 0 {
                // Insert the first key into the model, the slope doesn't matter
                add_spline_key(&key, make_model_position(offset, 0.0));
                last_spline_key = key;
                last_spline_offset = offset;
                slope_upper_bound = Slope::MAX;
                slope_lower_bound = Slope::MIN;
            }
            else if offset == segment.read().dentries.len() - 1 {
                // Insert the last key into the model, the slope is calculated based on the last spline key
                let cur_slope = Self::calculate_slope(&last_spline_key, &key, last_spline_offset, offset);
                add_spline_key(&key, make_model_position(offset, cur_slope));
            }
            else {
                // Calculate the slope between the last spline key and the current key
                let cur_slope = Self::calculate_slope(&last_spline_key, &key, last_spline_offset, offset);
                if cur_slope >= slope_upper_bound || cur_slope <= slope_lower_bound {
                    // If the slope is out of bounds, we need to insert a new spline key(last key)
                    let temp_slope = Self::calculate_slope(&last_spline_key, &last_key, last_spline_offset, last_offset);
                    add_spline_key(&last_key, make_model_position(offset, temp_slope));
                    last_spline_key = last_key;
                    last_spline_offset = last_offset;
                    slope_upper_bound = Self::calculate_slope(&last_key, &key, last_offset, offset + MAX_MODEL_ERROR as SegmentOffset);
                    slope_lower_bound = Self::calculate_slope(&last_key, &key, last_offset, offset - MAX_MODEL_ERROR as SegmentOffset);
                }
                else {
                    // Update the slope bounds
                    let temp_upper_slope = Self::calculate_slope(&last_spline_key, &key, last_spline_offset, offset + MAX_MODEL_ERROR as SegmentOffset);
                    if temp_upper_slope < slope_upper_bound {
                        slope_upper_bound = temp_upper_slope;
                    }
                    let temp_lower_slope = Self::calculate_slope(&last_spline_key, &key, last_spline_offset, offset - MAX_MODEL_ERROR as SegmentOffset);
                    if temp_lower_slope > slope_lower_bound {
                        slope_lower_bound = temp_lower_slope;
                    }
                }
            }
            last_key = key;
            last_offset = offset;
            offset += 1;
        }
        Ok(())
    }

    fn load_ge(&self, key: &ModelKey) -> Option<(ModelKey, Arc<ModelPosition>)> {
        let locked = self.0.lock();
        let mut cursor = locked.cursor(key.0);

        let cur_node = cursor.load();
        if cur_node.is_some() {
            return Some((ModelKey(cursor.index()), cur_node.unwrap().clone()));
        }

        cursor.next();
        let next_node = cursor.load();
        if next_node.is_none() {
            return None;
        }
        return Some((ModelKey(cursor.index()), next_node.unwrap().clone()));
    }

    pub(super) fn predict_position(&self, name: &str) -> Result<ModelPosition> {
        let key = ModelKey::from_name(name);
        let pos_ge = self.load_ge(&key);
        if let Some((next_key, next_pos)) = pos_ge {
            if key == next_key {
                return Ok(ModelPosition {
                    segment: next_pos.segment.clone(),
                    offset: next_pos.offset,
                    slope: 0.0,
                });
            }
            if next_pos.offset == 0 {
                return_errno!(Errno::ENOENT);
            }
            return Ok(ModelPosition {
                segment: next_pos.segment.clone(),
                offset: Self::linear_interpolation(&key, &next_key, next_pos.offset, &next_pos.slope),
                slope: 0.0,
            });
        }
        return_errno!(Errno::ENOENT);
    }

    pub(super) fn remove(&mut self, segment: Arc<RwLock<LearnedSegment>>) -> Result<()> {
        let mut locked = self.0.lock();
        for entry in segment.read().dentries.iter() {
            let key = ModelKey::from_name(&entry.name);
            locked.remove(key.0);
        }
        Ok(())
    }
}