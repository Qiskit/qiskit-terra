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

// ParameterExpressionExpression class using symengine C wrapper interface

use std::convert::From;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Neg};
use hashbrown::HashMap;

use crate::symbol_expr::{SymbolExpr, Value};
use crate::symbol_parser::parse_expression;

use num_complex::Complex64;

#[derive(Debug)]
pub struct ParameterExpression {
    expr_: SymbolExpr,
}

impl ParameterExpression {
    pub fn default() -> Self {
        Self {
            expr_: SymbolExpr::Value( Value::Real(0.0)),
        }
    }

    pub fn new(expr: &str) -> Self {
        Self {
            expr_: parse_expression(expr),
        }
    }

    pub fn to_string(&self) -> String {
        self.expr_.to_string()
    }

    pub fn num_symbols(&self) -> usize {
        self.expr_.symbols().len()
    }

    pub fn symbols(&self) -> Vec<String> {
        let mut symbols: Vec<String> = self
            .expr_.symbols()
            .iter()
            .map(|key| key.to_string())
            .collect();
        symbols.sort();
        symbols
    }
    pub fn has_symbol(&self, symbol: String) -> bool {
        self.expr_.symbols().contains(&symbol)
    }

    pub fn bind(&mut self, map: &HashMap<String, Value>) -> ParameterExpression {
        ParameterExpression{expr_: self.expr_.bind(&map)}
    }

    pub fn subs(&self, map: &HashMap<String, ParameterExpression>) -> ParameterExpression {
        let subs_map : HashMap::<String, SymbolExpr> = 
        map.iter().map(|(key, val)| (key.clone(), val.expr_.clone())).collect();
        ParameterExpression{expr_: self.expr_.subs(&subs_map)}
    }

    pub fn real(&self) -> Option<f64> {
        self.expr_.real()
    }
    pub fn imag(&self) -> Option<f64> {
        self.expr_.imag()
    }
    pub fn complex(&self) -> Option<Complex64> {
        self.expr_.complex()
    }

    pub fn is_complex(&self) -> bool {
        self.expr_.is_complex()
    }
    pub fn is_real(&self) -> bool {
        self.expr_.is_real()
    }

    fn add_expr(self, rhs: Self) -> Self {
        Self {
            expr_: self.expr_ + rhs.expr_,
        }
    }

    fn add_assign_expr(&mut self, rhs: Self) {
        self.expr_ = self.expr_.clone() + rhs.expr_;
    }

    fn sub_expr(self, rhs: Self) -> Self {
        Self {
            expr_: self.expr_ - rhs.expr_,
        }
    }

    fn sub_assign_expr(&mut self, rhs: Self) {
        self.expr_ = self.expr_.clone() - rhs.expr_;
    }

    fn mul_expr(self, rhs: Self) -> Self {
        Self {
            expr_: self.expr_ * rhs.expr_,
        }
    }

    fn mul_assign_expr(&mut self, rhs: Self) {
        self.expr_ = self.expr_.clone() * rhs.expr_;
    }

    fn div_expr(self, rhs: Self) -> Self {
        Self {
            expr_: self.expr_ / rhs.expr_,
        }
    }

    fn div_assign_expr(&mut self, rhs: Self) {
        self.expr_ = self.expr_.clone() / rhs.expr_;
    }

    pub fn sin(self) -> Self {
        Self {
            expr_: self.expr_.sin(),
        } 
    }

    pub fn cos(self) -> Self {
        Self {
            expr_: self.expr_.cos(),
        } 
    }

    pub fn tan(self) -> Self {
        Self {
            expr_: self.expr_.tan(),
        } 
    }

    pub fn arcsin(self) -> Self {
        Self {
            expr_: self.expr_.asin(),
        } 
    }

    pub fn arccos(self) -> Self {
        Self {
            expr_: self.expr_.acos(),
        } 
    }

    pub fn arctan(self) -> Self {
        Self {
            expr_: self.expr_.atan(),
        } 
    }

    pub fn exp(self) -> Self {
        Self {
            expr_: self.expr_.exp(),
        } 
    }

    pub fn log(self) -> Self {
        Self {
            expr_: self.expr_.log(),
        } 
    }

    pub fn abs(self) -> Self {
        Self {
            expr_: self.expr_.abs(),
        } 
    }

    pub fn pow<T: Into<Self>>(self, prm: T) -> Self {
        let t : ParameterExpression = prm.into();
        Self {expr_: self.expr_.pow(&t.expr_),}
    }
}

impl Clone for ParameterExpression {
    fn clone(&self) -> Self {
        Self {
            expr_: self.expr_.clone()
        }
    }
}

impl PartialEq for ParameterExpression {
    fn eq(&self, rprm: &Self) -> bool {
        self.expr_ == rprm.expr_
    }
}


// =============================
// Make from Rust native types
// =============================

impl From<i32> for ParameterExpression {
    fn from(v: i32) -> Self {
        Self {
            expr_: SymbolExpr::Value( Value::Real(v as f64)),
        }
    }
}
impl From<i64> for ParameterExpression {
    fn from(v: i64) -> Self {
        Self {
            expr_: SymbolExpr::Value( Value::Real(v as f64)),
        }
    }
}

impl From<u32> for ParameterExpression {
    fn from(v: u32) -> Self {
        Self {
            expr_: SymbolExpr::Value( Value::Real(v as f64)),
        }
    }
}

impl From<f64> for ParameterExpression {
    fn from(v: f64) -> Self {
        Self {
            expr_: SymbolExpr::Value( Value::Real(v)),
        }
    }
}

impl From<Complex64> for ParameterExpression {
    fn from(v: Complex64) -> Self {
        Self {
            expr_: SymbolExpr::Value( Value::Complex(v)),
        }
    }
}

impl From<&str> for ParameterExpression {
    fn from(v: &str) -> Self {
        Self::new(v)
    }
}

// =============================
// Unary operations
// =============================
impl Neg for ParameterExpression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            expr_: -self.expr_,
        }
    }
}

// =============================
// Add
// =============================

macro_rules! add_impl_expr {
    ($($t:ty)*) => ($(
        impl Add<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn add(self, other: $t) -> Self::Output {
                self.add_expr(other.into())
            }
        }

        impl AddAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn add_assign(&mut self, other: $t) {
                self.add_assign_expr(other.into())
            }
        }
    )*)
}

add_impl_expr! {f64 i32 u32 ParameterExpression}

// =============================
// Sub
// =============================

macro_rules! sub_impl_expr {
    ($($t:ty)*) => ($(
        impl Sub<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn sub(self, other: $t) -> Self::Output {
                self.sub_expr(other.into())
            }
        }

        impl SubAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn sub_assign(&mut self, other: $t) {
                self.sub_assign_expr(other.into())
            }
        }
    )*)
}

sub_impl_expr! {f64 i32 u32 ParameterExpression}

// =============================
// Mul
// =============================

macro_rules! mul_impl_expr {
    ($($t:ty)*) => ($(
        impl Mul<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn mul(self, other: $t) -> Self::Output {
                self.mul_expr(other.into())
            }
        }

        impl MulAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn mul_assign(&mut self, other: $t) {
                self.mul_assign_expr(other.into())
            }
        }
    )*)
}

mul_impl_expr! {f64 i32 u32 ParameterExpression}

// =============================
// Div
// =============================

macro_rules! div_impl_expr {
    ($($t:ty)*) => ($(
        impl Div<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn div(self, other: $t) -> Self::Output {
                self.div_expr(other.into())
            }
        }

        impl DivAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn div_assign(&mut self, other: $t) {
                self.div_assign_expr(other.into())
            }
        }
    )*)
}

div_impl_expr! {f64 i32 u32 ParameterExpression}
