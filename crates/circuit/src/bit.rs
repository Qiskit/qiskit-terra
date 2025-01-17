use std::fmt::Debug;

/// Keeps information about where a qubit is located within the circuit.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct BitInfo {
    register_idx: u32,
    index: u32,
}

impl BitInfo {
    pub fn new(register_idx: u32, index: u32) -> Self {
        Self {
            register_idx,
            index,
        }
    }

    pub fn register_index(&self) -> u32 {
        self.register_idx
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}
