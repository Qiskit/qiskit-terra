// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ahash::RandomState;
use indexmap::{
    map::{IntoIter as BaseIntoIter, Iter as BaseIter, Keys as BaseKeys, Values as BaseValues},
    IndexMap,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPy;
use rustworkx_core::dictmap::InitWithHasher;
use std::ops::Index;
use std::{hash::Hash, mem::swap};

type BaseMap<K, V> = IndexMap<K, V, RandomState>;

///
/// An `IndexMap`-like structure thet can be used when one of the keys can have a `None` value.
///
/// This structure is essentially a wrapper around the `IndexMap<K, V>` struct that allows the
/// storage of `Option<K>` key values as `K`` and keep an extra slot reserved only for the
/// `None` instance. There are some upsides to this including:
///
/// The ability to index using Option<&K> to index a specific key.
/// Store keys as non option wrapped to obtain references to K instead of reference to Option<K>.
///
/// **Warning:** This is an experimental feature and should be used with care as it does not
/// fully implement all the methods present in `IndexMap<K, V>` due to API limitations.
#[derive(Debug, Clone)]
pub(crate) struct NullableIndexMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    map: BaseMap<K, V>,
    null_val: Option<V>,
}

impl<K, V> NullableIndexMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Returns a reference to the value stored at `key`, if it does not exist
    /// `None` is returned instead.
    pub fn get(&self, key: Option<&K>) -> Option<&V> {
        match key {
            Some(key) => self.map.get(key),
            None => self.null_val.as_ref(),
        }
    }

    /// Returns a mutable reference to the value stored at `key`, if it does not
    /// exist `None` is returned instead.
    pub fn get_mut(&mut self, key: Option<&K>) -> Option<&mut V> {
        match key {
            Some(key) => self.map.get_mut(key),
            None => self.null_val.as_mut(),
        }
    }

    /// Inserts a `value` in the slot alotted to `key`.
    ///
    /// If a previous value existed there previously it will be returned, otherwise
    /// `None` will be returned.
    pub fn insert(&mut self, key: Option<K>, value: V) -> Option<V> {
        match key {
            Some(key) => self.map.insert(key, value),
            None => {
                let mut old_val = Some(value);
                swap(&mut old_val, &mut self.null_val);
                old_val
            }
        }
    }

    /// Creates an instance of `NullableIndexMap<K, V>` with capacity to hold `n`+1 key-value
    /// pairs.
    ///
    /// Notice that an extra space needs to be alotted to store the instance of `None` a
    /// key.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            map: BaseMap::with_capacity(n),
            null_val: None,
        }
    }

    /// Creates an instance of `NullableIndexMap<K, V>` from an iterator over instances of
    /// `(Option<K>, V)`.
    pub fn from_iter<'a, I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Option<K>, V)> + 'a,
    {
        let mut null_val = None;
        let filtered = iter.into_iter().filter_map(|item| match item {
            (Some(key), value) => Some((key, value)),
            (None, value) => {
                null_val = Some(value);
                None
            }
        });
        Self {
            map: IndexMap::from_iter(filtered),
            null_val,
        }
    }

    /// Returns `true` if the map contains a slot indexed by `key`, otherwise `false`.
    pub fn contains_key(&self, key: Option<&K>) -> bool {
        match key {
            Some(key) => self.map.contains_key(key),
            None => self.null_val.is_some(),
        }
    }

    /// Extends the key-value pairs in the map with the contents of an iterator over
    /// `(Option<K>, V)`.
    ///
    /// If an already existent key is provided, it will be replaced by the entry provided
    /// in the iterator.
    pub fn extend<'a, I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (Option<K>, V)> + 'a,
    {
        let filtered = iter.into_iter().filter_map(|item| match item {
            (Some(key), value) => Some((key, value)),
            (None, value) => {
                self.null_val = Some(value);
                None
            }
        });
        self.map.extend(filtered)
    }

    /// Removes the entry allotted to `key` from the map and returns it. The index of
    /// this entry is then replaced by the entry located at the last index.
    ///
    /// `None` will be returned if the `key` is not present in the map.
    pub fn swap_remove(&mut self, key: Option<&K>) -> Option<V> {
        match key {
            Some(key) => self.map.swap_remove(key),
            None => {
                let mut ret_val = None;
                swap(&mut ret_val, &mut self.null_val);
                ret_val
            }
        }
    }

    /// Returns an iterator over references of the key-value pairs of the map.
    // TODO: Remove once `NullableIndexMap` is being consumed.
    #[allow(dead_code)]
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            map: self.map.iter(),
            null_value: self.null_val.as_ref(),
        }
    }

    /// Returns an iterator over references of the keys present in the map.
    pub fn keys(&self) -> Keys<K, V> {
        Keys {
            map_keys: self.map.keys(),
            null_value: self.null_val.is_some(),
        }
    }

    /// Returns an iterator over references of all the values present in the map.
    pub fn values(&self) -> Values<K, V> {
        Values {
            map_values: self.map.values(),
            null_value: &self.null_val,
        }
    }

    /// Returns the number of key-value pairs present in the map.
    pub fn len(&self) -> usize {
        self.map.len() + self.null_val.is_some() as usize
    }
}

impl<K, V> IntoIterator for NullableIndexMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    type Item = (Option<K>, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            map: self.map.into_iter(),
            null_value: self.null_val,
        }
    }
}

/// Iterator for the key-value pairs in `NullableIndexMap`.
pub struct Iter<'a, K, V> {
    map: BaseIter<'a, K, V>,
    null_value: Option<&'a V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (Option<&'a K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((key, val)) = self.map.next() {
            Some((Some(key), val))
        } else {
            self.null_value.take().map(|value| (None, value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map.size_hint().0 + self.null_value.is_some() as usize,
            self.map
                .size_hint()
                .1
                .map(|hint| hint + self.null_value.is_some() as usize),
        )
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.map.len() + self.null_value.is_some() as usize
    }
}

/// Owned iterator over the key-value pairs in `NullableIndexMap`.
pub struct IntoIter<K, V>
where
    V: Clone,
{
    map: BaseIntoIter<K, V>,
    null_value: Option<V>,
}

impl<K, V> Iterator for IntoIter<K, V>
where
    V: Clone,
{
    type Item = (Option<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((key, val)) = self.map.next() {
            Some((Some(key), val))
        } else if self.null_value.is_some() {
            let mut value = None;
            swap(&mut value, &mut self.null_value);
            Some((None, value.unwrap()))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map.size_hint().0 + self.null_value.is_some() as usize,
            self.map
                .size_hint()
                .1
                .map(|hint| hint + self.null_value.is_some() as usize),
        )
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V>
where
    V: Clone,
{
    fn len(&self) -> usize {
        self.map.len() + self.null_value.is_some() as usize
    }
}

/// Iterator over the keys of a `NullableIndexMap`.
pub struct Keys<'a, K, V> {
    map_keys: BaseKeys<'a, K, V>,
    null_value: bool,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = Option<&'a K>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(key) = self.map_keys.next() {
            Some(Some(key))
        } else if self.null_value {
            self.null_value = false;
            Some(None)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map_keys.size_hint().0 + self.null_value as usize,
            self.map_keys
                .size_hint()
                .1
                .map(|hint| hint + self.null_value as usize),
        )
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    fn len(&self) -> usize {
        self.map_keys.len() + self.null_value as usize
    }
}

/// Iterator over the values of a `NullableIndexMap`.
pub struct Values<'a, K, V> {
    map_values: BaseValues<'a, K, V>,
    null_value: &'a Option<V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(value) = self.map_values.next() {
            Some(value)
        } else if self.null_value.is_some() {
            let return_value = self.null_value;
            self.null_value = &None;
            return_value.as_ref()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map_values.size_hint().0 + self.null_value.is_some() as usize,
            self.map_values
                .size_hint()
                .1
                .map(|hint| hint + self.null_value.is_some() as usize),
        )
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    fn len(&self) -> usize {
        self.map_values.len() + self.null_value.is_some() as usize
    }
}

impl<K, V> Index<Option<&K>> for NullableIndexMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    type Output = V;
    fn index(&self, index: Option<&K>) -> &Self::Output {
        match index {
            Some(k) => self.map.index(k),
            None => match &self.null_val {
                Some(val) => val,
                None => panic!("The provided key is not present in map: None"),
            },
        }
    }
}

impl<K, V> Default for NullableIndexMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self {
            map: IndexMap::default(),
            null_val: None,
        }
    }
}

impl<'py, K, V> FromPyObject<'py> for NullableIndexMap<K, V>
where
    K: IntoPy<PyObject> + FromPyObject<'py> + Eq + Hash + Clone,
    V: IntoPy<PyObject> + FromPyObject<'py> + Clone,
{
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let map: IndexMap<Option<K>, V, RandomState> = ob.extract()?;
        let mut null_val: Option<V> = None;
        let filtered = map
            .into_iter()
            .filter_map(|(key, value)| match (key, value) {
                (Some(key), value) => Some((key, value)),
                (None, value) => {
                    null_val = Some(value);
                    None
                }
            });
        Ok(Self {
            map: filtered.collect(),
            null_val,
        })
    }
}

impl<K, V> IntoPy<PyObject> for NullableIndexMap<K, V>
where
    K: IntoPy<PyObject> + Eq + Hash + Clone,
    V: IntoPy<PyObject> + Clone,
{
    fn into_py(self, py: Python<'_>) -> PyObject {
        let map_object = self.map.into_py(py);
        let bound_map_obj = map_object.bind(py);
        let downcast_dict: &Bound<PyDict> = bound_map_obj.downcast().unwrap();
        if let Some(null_val) = self.null_val {
            downcast_dict
                .set_item(py.None(), null_val.into_py(py))
                .unwrap();
        }
        map_object
    }
}

impl<K, V> ToPyObject for NullableIndexMap<K, V>
where
    K: ToPyObject + Eq + Hash + Clone,
    V: ToPyObject + Clone,
{
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let map_object = self.map.to_object(py);
        let bound_map_obj = map_object.bind(py);
        let downcast_dict: &Bound<PyDict> = bound_map_obj.downcast().unwrap();
        if let Some(null_val) = &self.null_val {
            downcast_dict
                .set_item(py.None(), null_val.to_object(py))
                .unwrap();
        }
        map_object
    }
}
