// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use indexmap::IndexSet;

/// A key to retrieve a value (by reference) from an interner of the same type.  This is narrower
/// than a true reference, at the cost that it is explicitly not lifetime bound to the interner it
/// came from; it is up to the user to ensure that they never attempt to query an interner with a
/// key from a different interner.
#[derive(Debug, Eq, PartialEq)]
pub struct Interned<T: ?Sized> {
    index: u32,
    // Storing the type of the interned value adds a small amount more type safety to the interner
    // keys when there's several interners in play close to each other.  We use `*const T` because
    // the `Interned value` is like a non-lifetime-bound reference to data stored in the interner;
    // `Interned` doesn't own the data (which would be implied by `T`), and it's not using the
    // static lifetime system (which would be implied by `&'_ T`, and require us to propagate the
    // lifetime bound).
    _type: PhantomData<*const T>,
}
// The `PhantomData` marker prevents various useful things from being derived (for `Clone` and
// `Copy` it's an awkward effect of the derivation system), so we have manual implementations.
impl<T: ?Sized> Clone for Interned<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized> Copy for Interned<T> {}
unsafe impl<T: ?Sized> Send for Interned<T> {}
unsafe impl<T: ?Sized> Sync for Interned<T> {}

/// An append-only data structure for interning generic Rust types.
///
/// The interner can lookup keys using a reference type, and will create the corresponding owned
/// allocation on demand, if a matching entry is not already stored.  It returns manual keys into
/// itself (the `Interned` type), rather than raw references; the `Interned` type is narrower than a
/// true reference.
///
/// This is only implemented for owned types that implement `Default`, so that the convenience
/// method `Interner::get_default` can work reliably and correctly; the "default" index needs to be
/// guaranteed to be reserved and present for safety.
///
/// # Examples
///
/// ```rust
/// let mut interner = Interner::<[usize]>::new();
///
/// // These are of type `Interned<[usize]>`.
/// let default_empty = interner.get_default();
/// let empty = interner.insert(&[]);
/// let other_empty = interner.insert(&[]);
/// let key = interner.insert(&[0, 1, 2, 3, 4]);
///
/// assert_eq!(empty, other_empty);
/// assert_eq!(empty, default_empty);
/// assert_ne!(empty, key);
///
/// assert_eq!(interner.get(empty), &[]);
/// assert_eq!(interner.get(key), &[0, 1, 2, 3, 4]);
/// ```
#[derive(Default)]
pub struct Interner<T: ?Sized + ToOwned>(IndexSet<<T as ToOwned>::Owned, ::ahash::RandomState>);

// `Clone` and `Debug` can't use the derivation mechanism because the values that are actually
// stored are of type `<T as ToOwned>::Owned`, which the derive system doesn't reason about.
impl<T> Clone for Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> fmt::Debug for Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_tuple("Interner").field(&self.0).finish()
    }
}

impl<T> Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: Hash + Eq + Default,
{
    /// Construct a new interner.  The stored type must have a default value, in order for
    /// `Interner::get_default` to reliably work correctly without a hash lookup (though ideally
    /// we'd just use specialisation to do that).
    pub fn new() -> Self {
        Self::with_capacity(1)
    }

    /// Retrieve the key corresponding to the default store, without any hash or equality lookup.
    /// For example, if the interned type is `[Clbit]`, the default key corresponds to the empty
    /// slice `&[]`.  This is a common operation with the cargs interner, for things like pushing
    /// gates.
    ///
    /// In an ideal world, we wouldn't have the `Default` trait bound on `new`, but would use
    /// specialisation to insert the default key only if the stored value implemented `Default`
    /// (we'd still trait-bound this method).
    #[inline(always)]
    pub fn get_default(&self) -> Interned<T> {
        Interned {
            index: 0,
            _type: PhantomData,
        }
    }

    /// Create an interner with enough space to hold `capacity` entries.
    ///
    /// Note that the default item of the interner is always allocated and given a key immediately,
    /// which will use one slot of the capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut set = IndexSet::with_capacity_and_hasher(capacity, ::ahash::RandomState::new());
        set.insert(Default::default());
        Self(set)
    }
}

impl<T> Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: Hash + Eq,
{
    /// Retrieve a reference to the stored value for this key.
    pub fn get(&self, index: Interned<T>) -> &T {
        self.0
            .get_index(index.index as usize)
            .expect(
                "the caller is responsible for only using interner keys from the correct interner",
            )
            .borrow()
    }

    /// Internal worker function that inserts an owned value assuming that the value didn't
    /// previously exist in the map.
    fn insert_new(&mut self, value: <T as ToOwned>::Owned) -> u32 {
        let index = self.0.len();
        if index == u32::MAX as usize {
            panic!("interner is out of space");
        }
        let _inserted = self.0.insert(value);
        debug_assert!(_inserted);
        index as u32
    }

    /// Get an interner key corresponding to the given referenced type.  If not already stored, this
    /// function will allocate a new owned value to use as the storage.
    ///
    /// If you already have an owned value, use `insert_owned`, but in general this function will be
    /// more efficient *unless* you already had the value for other reasons.
    pub fn insert(&mut self, value: &T) -> Interned<T>
    where
        T: Hash + Eq,
    {
        let index = match self.0.get_index_of(value) {
            Some(index) => index as u32,
            None => self.insert_new(value.to_owned()),
        };
        Interned {
            index,
            _type: PhantomData,
        }
    }

    /// Get an interner key corresponding to the given owned type.  If not already stored, the value
    /// will be used as the key, otherwise it will be dropped.
    ///
    /// If you don't already have the owned value, use `insert`; this will only allocate if the
    /// lookup fails.
    pub fn insert_owned(&mut self, value: <T as ToOwned>::Owned) -> Interned<T> {
        let index = match self.0.get_index_of(&value) {
            Some(index) => index as u32,
            None => self.insert_new(value),
        };
        Interned {
            index,
            _type: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn default_key_exists() {
        let mut interner = Interner::<[u32]>::new();
        assert_eq!(interner.get_default(), interner.get_default());
        assert_eq!(interner.get(interner.get_default()), &[]);
        assert_eq!(interner.insert_owned(Vec::new()), interner.get_default());
        assert_eq!(interner.insert(&[]), interner.get_default());

        let capacity = Interner::<str>::with_capacity(4);
        assert_eq!(capacity.get_default(), capacity.get_default());
        assert_eq!(capacity.get(capacity.get_default()), "");
    }
}
