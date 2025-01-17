use indexmap::IndexSet;
use pyo3::{exceptions::PyTypeError, intern, types::PyAnyMethods, FromPyObject};
use std::{
    hash::{Hash, Hasher},
    ops::Index,
    sync::Mutex,
};

use crate::{
    imports::{CLASSICAL_REGISTER, QUANTUM_REGISTER, REGISTER},
    Clbit, Qubit,
};

/// This represents the hash value of a Register according to the register's
/// name and number of qubits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegisterAsKey {
    Register((String, u32)),
    Quantum((String, u32)),
    Classical((String, u32)),
}

impl RegisterAsKey {
    #[inline]
    pub fn reduce(&self) -> (u32, &str) {
        match self {
            RegisterAsKey::Register(key) => (key.1, key.0.as_str()),
            RegisterAsKey::Quantum(key) => (key.1, key.0.as_str()),
            RegisterAsKey::Classical(key) => (key.1, key.0.as_str()),
        }
    }

    #[inline]
    pub fn name(&self) -> &str {
        match self {
            RegisterAsKey::Register(key) => key.0.as_str(),
            RegisterAsKey::Quantum(key) => key.0.as_str(),
            RegisterAsKey::Classical(key) => key.0.as_str(),
        }
    }

    #[inline]
    pub fn index(&self) -> u32 {
        match self {
            RegisterAsKey::Register(key) => key.1,
            RegisterAsKey::Quantum(key) => key.1,
            RegisterAsKey::Classical(key) => key.1,
        }
    }

    #[inline]
    pub fn type_identifier(&self) -> &str {
        match self {
            RegisterAsKey::Register(_) => "Register",
            RegisterAsKey::Quantum(_) => "QuantumRegister",
            RegisterAsKey::Classical(_) => "ClassicalRegister",
        }
    }
}

impl<'py> FromPyObject<'py> for RegisterAsKey {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        if ob.is_instance(REGISTER.get_bound(ob.py()))? {
            let (name, num_qubits) = (
                ob.getattr(intern!(ob.py(), "name"))?.extract()?,
                ob.len()? as u32,
            );
            if ob.is_instance(CLASSICAL_REGISTER.get_bound(ob.py()))? {
                return Ok(RegisterAsKey::Classical((name, num_qubits)));
            } else if ob.is_instance(QUANTUM_REGISTER.get_bound(ob.py()))? {
                return Ok(RegisterAsKey::Quantum((name, num_qubits)));
            } else {
                return Ok(RegisterAsKey::Register((name, num_qubits)));
            }
        }
        Err(PyTypeError::new_err(
            "The provided argument was not a register.",
        ))
    }
}
/// Described the desired behavior of a Register.
pub trait Register {
    /// The type of bit stored by the [Register]
    type Bit;

    /// Returns the size of the [Register].
    fn len(&self) -> usize;
    /// Checks if the [Register] is empty.
    fn is_empty(&self) -> bool;
    /// Returns the name of the [Register].
    fn name(&self) -> &str;
    /// Checks if a bit exists within the [Register].
    fn contains(&self, bit: Self::Bit) -> bool;
    /// Finds the local index of a certain bit within [Register].
    fn find_index(&self, bit: Self::Bit) -> Option<u32>;
    /// Return an iterator over all the bits in the register
    fn bits(&self) -> impl ExactSizeIterator<Item = Self::Bit>;
    /// Returns the register as a Key
    fn as_key(&self) -> &RegisterAsKey;
}

macro_rules! create_register {
    ($name:ident, $bit:ty, $counter:ident, $prefix:literal, $key:expr) => {
        static $counter: Mutex<u32> = Mutex::new(0);

        #[derive(Debug, Clone, Eq)]
        pub struct $name {
            register: IndexSet<<$name as Register>::Bit>,
            key: RegisterAsKey,
        }

        impl $name {
            pub fn new(size: Option<usize>, name: Option<String>, bits: Option<&[$bit]>) -> Self {
                let register: IndexSet<<$name as Register>::Bit> = if let Some(size) = size {
                    (0..size).map(|bit| <$bit>::new(bit)).collect()
                } else if let Some(bits) = bits {
                    bits.iter().copied().collect()
                } else {
                    panic!("You should only provide either a size or the bit indices, not both.")
                };
                let name = if let Some(name) = name {
                    name
                } else {
                    let count = if let Ok(ref mut count) = $counter.try_lock() {
                        let curr = **count;
                        **count += 1;
                        curr
                    } else {
                        panic!("Could not access register counter.")
                    };
                    format!("{}{}", $prefix, count)
                };
                let length: u32 = register.len().try_into().unwrap();
                Self {
                    register,
                    key: $key((name, length)),
                }
            }
        }

        impl Register for $name {
            type Bit = $bit;

            fn len(&self) -> usize {
                self.register.len()
            }

            fn is_empty(&self) -> bool {
                self.register.is_empty()
            }

            fn name(&self) -> &str {
                self.key.name()
            }

            fn contains(&self, bit: Self::Bit) -> bool {
                self.register.contains(&bit)
            }

            fn find_index(&self, bit: Self::Bit) -> Option<u32> {
                self.register.get_index_of(&bit).map(|idx| idx as u32)
            }

            fn bits(&self) -> impl ExactSizeIterator<Item = Self::Bit> {
                self.register.iter().copied()
            }

            fn as_key(&self) -> &RegisterAsKey {
                &self.key
            }
        }

        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                (self.key).hash(state);
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.register.len() == other.register.len() && self.key == other.key
            }
        }

        impl Index<usize> for $name {
            type Output = $bit;

            fn index(&self, index: usize) -> &Self::Output {
                self.register.index(index)
            }
        }

        impl From<(usize, Option<String>)> for $name {
            fn from(value: (usize, Option<String>)) -> Self {
                Self::new(Some(value.0), value.1, None)
            }
        }

        impl From<&[$bit]> for $name {
            fn from(value: &[$bit]) -> Self {
                Self::new(None, None, Some(value))
            }
        }

        impl From<(&[$bit], String)> for $name {
            fn from(value: (&[$bit], String)) -> Self {
                Self::new(None, Some(value.1), Some(value.0))
            }
        }
    };
}

create_register!(
    QuantumRegister,
    Qubit,
    QREG_COUNTER,
    "qr",
    RegisterAsKey::Quantum
);

create_register!(
    ClassicalRegister,
    Clbit,
    CREG_COUNTER,
    "cr",
    RegisterAsKey::Classical
);
