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

use std::sync::Arc;
use std::ops::{Add, Div, Mul, Sub, Neg};
use std::convert::From;
use hashbrown::{HashMap, HashSet};

use num_complex::Complex64;

#[derive(Debug, Clone)]
pub enum SymbolExpr {
    Symbol(Symbol),
    Value(Value),
    Unary(Arc<Unary>),
    Binary(Arc<Binary>),
}

#[derive(Debug, Clone)]
pub struct Symbol {
    name : String,
}

// ================================
// real number and complex number
// (separate for performance)
// ================================
#[derive(Debug, Clone)]
pub enum Value {
    Real(f64),
    Complex(Complex64),
}

// ================================
// Operators
// ================================
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOps {
    Abs,
    Neg,
    Sin,
    Asin,
    Cos,
    Acos,
    Tan,
    Atan,
    Exp,
    Log,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOps {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone)]
pub struct Unary {
    op : UnaryOps,
    expr : SymbolExpr,
}

#[derive(Debug, Clone)]
pub struct Binary {
    op : BinaryOps,
    lhs : SymbolExpr,
    rhs : SymbolExpr,
}

impl SymbolExpr {
    pub fn to_string(&self) -> String {
        match self {
            SymbolExpr::Symbol(e) => e.to_string(),
            SymbolExpr::Value(e) => e.to_string(),
            SymbolExpr::Unary(e) => e.to_string(),
            SymbolExpr::Binary(e) => e.to_string(),
        }
    }

    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => e.bind(maps),
            SymbolExpr::Value(e) => SymbolExpr::Value(e.clone()),
            SymbolExpr::Unary(e) => e.bind(maps),
            SymbolExpr::Binary(e) => e.bind(maps),
        }
    }

    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => e.subs(maps),
            SymbolExpr::Value(e) => SymbolExpr::Value(e.clone()),
            SymbolExpr::Unary(e) => e.subs(maps),
            SymbolExpr::Binary(e) => e.subs(maps),
        }
    }

    pub fn eval(&self, recurse: bool) -> Option<Value> {
        match self {
            SymbolExpr::Symbol(_) => None,
            SymbolExpr::Value(e) => Some(e.clone()),
            SymbolExpr::Unary(e) => e.eval(recurse),
            SymbolExpr::Binary(e) => e.eval(recurse),
        }
    }

    pub fn derivative(&self, param: &SymbolExpr) -> SymbolExpr {
        if self == param {
            SymbolExpr::Value( Value::Real(1.0))
        } else {
            match self {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => SymbolExpr::Value( Value::Real(0.0)),
                SymbolExpr::Unary(e) => e.derivative(param),
                SymbolExpr::Binary(e) => e.derivative(param),
            }
        }
    }

    pub fn sign(&self) -> f64 {
        match self.eval(true) {
            Some(v) => if v.as_real() > 0.0 {
                1.0
            } else if v.as_real() < 0.0 {
                -1.0
            } else {
                0.0
            },
            None => match self {
                SymbolExpr::Symbol(_) => 1.0,
                SymbolExpr::Value(e) => match e {
                    Value::Real(r) => if *r > 0.0 {
                        return 1.0;
                    } else if *r < 0.0 {
                        return -1.0;
                    } else {
                        return 0.0;
                    },
                    Value::Complex(_) => 0.0,
                },
                SymbolExpr::Unary(e) => e.sign(),
                SymbolExpr::Binary(e) => e.sign(),
            },
        }
    }

    pub fn real(&self) -> Option<f64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r),
                Value::Complex(c) => Some(c.re),
            }
            None => None,
        }
    }
    pub fn imag(&self) -> Option<f64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(_) => Some(0.0),
                Value::Complex(c) => Some(c.im),
            }
            None => None,
        }
    }
    pub fn complex(&self) -> Option<Complex64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(_) => Some(0.0.into()),
                Value::Complex(c) => Some(c),
            }
            None => None,
        }
    }

    pub fn symbols(&self) -> HashSet<String> {
        match self {
            SymbolExpr::Symbol(e) => HashSet::<String>::from([e.name.clone()]),
            SymbolExpr::Value(_) => HashSet::<String>::new(),
            SymbolExpr::Unary(e) => e.symbols(),
            SymbolExpr::Binary(e) => e.symbols(),
        }
    }

    pub fn has_symbol(&self, param: &String) -> bool {
        match self {
            SymbolExpr::Symbol(e) => e.name == *param,
            SymbolExpr::Value(_) => false,
            SymbolExpr::Unary(e) => e.has_symbol(param),
            SymbolExpr::Binary(e) => e.has_symbol(param),
        }
    }

    pub fn rcp(self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => SymbolExpr::Value( Value::Real(1.0)) / SymbolExpr::Symbol(e),
            SymbolExpr::Value(e) => SymbolExpr::Value( Value::Real(1.0)) / SymbolExpr::Value(e),
            SymbolExpr::Unary(e) => SymbolExpr::Value( Value::Real(1.0)) / SymbolExpr::Unary(e),
            SymbolExpr::Binary(ref e) => match e.op {
                BinaryOps::Div => SymbolExpr::Binary( Arc::new( Binary{ op: e.op.clone(), lhs: e.rhs.clone(), rhs: e.lhs.clone()}) ),
                _ => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: SymbolExpr::Value( Value::Real(1.0)), rhs: self.clone()}) ),
            }
        }
    }

    pub fn conjugate(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => SymbolExpr::Symbol(e.clone()),
            SymbolExpr::Value(e) => match e {
                Value::Complex(c) => SymbolExpr::Value( Value::Complex(c.conj())),
                _ => SymbolExpr::Value( e.clone()),
            },
            SymbolExpr::Unary(e) => SymbolExpr::Unary( Arc::new( Unary{ op: e.op.clone(), expr: e.expr.conjugate()}) ),
            SymbolExpr::Binary(e) => SymbolExpr::Binary( Arc::new( Binary{ op: e.op.clone(), lhs: e.lhs.conjugate(), rhs: e.rhs.conjugate()}) ),
        }
    }

    pub fn is_complex(&self) -> bool {
        match self {
            SymbolExpr::Symbol(_) => false,
            SymbolExpr::Value(e) => match e {
                Value::Complex(_) => true,
                _ => false,
            },
            SymbolExpr::Unary(e) => e.expr.is_complex(),
            SymbolExpr::Binary(e) => e.lhs.is_complex() || e.rhs.is_complex(),
        }
    }
    pub fn is_real(&self) -> bool {
        match self {
            SymbolExpr::Symbol(_) => true,
            SymbolExpr::Value(e) => match e {
                Value::Real(_) => true,
                _ => false,
            },
            SymbolExpr::Unary(e) => e.expr.is_real(),
            SymbolExpr::Binary(e) => e.lhs.is_real() && e.rhs.is_real(),
        }
    }
    pub fn is_int(&self) -> bool {
        match self {
            SymbolExpr::Symbol(_) => false,
            SymbolExpr::Value(e) => match e {
                Value::Real(r) => {
                    let t = r - r.floor();
                    return t < f64::EPSILON && t > -f64::EPSILON;
                }
                Value::Complex(c) => {
                    if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                        let t = c.re - c.re.floor();
                        return t < f64::EPSILON && t > -f64::EPSILON;
                    } else {
                        return false;
                    }
                }
            },
            SymbolExpr::Unary(e) => e.expr.is_int(),
            SymbolExpr::Binary(e) => e.lhs.is_int() && e.rhs.is_int(),
        }
    }

    pub fn abs(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().abs()),
            SymbolExpr::Unary(e) => match e.op {
                UnaryOps::Abs | UnaryOps::Neg => e.expr.abs(),
                _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Abs, expr: self.clone()} )),
            },
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Abs, expr: self.clone()} )),
        }
    }
    pub fn sin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().sin()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Sin, expr: self.clone()} )),
        }
    }
    pub fn asin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().asin()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Asin, expr: self.clone()} )),
        }
    }
    pub fn cos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().cos()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Cos, expr: self.clone()} )),
        }
    }
    pub fn acos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().acos()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Acos, expr: self.clone()} )),
        }
    }
    pub fn tan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().tan()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Tan, expr: self.clone()} )),
        }
    }
    pub fn atan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().atan()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Atan, expr: self.clone()} )),
        }
    }
    pub fn exp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().exp()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Exp, expr: self.clone()} )),
        }
    }
    pub fn log(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( l.clone().log()),
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Log, expr: self.clone()} )),
        }
    }
    pub fn pow(&self, rhs: &SymbolExpr) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => SymbolExpr::Value( l.clone().pow(r.clone())),
                _ => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Pow, lhs: SymbolExpr::Value(l.clone()), rhs: rhs.clone()}) ),
            },
            _ => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Pow, lhs: self.clone(), rhs: rhs.clone()} )),
        }
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary(e) => e.add_opt(rhs),
            SymbolExpr::Binary(e) => e.add_opt(rhs),
            _ => match rhs {
                SymbolExpr::Binary(r) => r.add_opt(self),
                _ => None,
            },
        }
    }
    // Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary(e) => e.sub_opt(rhs),
            SymbolExpr::Binary(e) => e.sub_opt(rhs),
            _ => match rhs {
                SymbolExpr::Binary(r) => match r.sub_opt(self) {
                    Some(e) => Some(-e),
                    _ => None,
                },
                _ => None,
            },
        }
    }
    // Mul with heuristic optimization
    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary(_) => None,   //TO DO add this
            SymbolExpr::Binary(e) => e.mul_opt(rhs),
            _ => match rhs {
                SymbolExpr::Binary(r) => r.mul_opt(self),
                _ => None,
            },
        }
    }
    // Div with heuristic optimization
    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary(_) => None,   //TO DO add this
            SymbolExpr::Binary(e) => e.div_opt(rhs),
            _ => None,
        }
    }
}

impl Add for SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        &self + &rhs
    }
}

impl Add for &SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        if *self == 0.0 {
            rhs.clone()
        } else if *rhs == 0.0 {
            self.clone()
        } else if *self == *rhs {
            match self {
                SymbolExpr::Value(l) => SymbolExpr::Value(l * &Value::Real(2.0)),
                _ => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value( Value::Real(2.0)), rhs: self.clone()} )),
            }
        } else {
            match self {
                SymbolExpr::Value(l) => match rhs {
                    SymbolExpr::Value(r) => SymbolExpr::Value(l + r),
                    SymbolExpr::Binary(r) => match r.add_opt(self) {
                        Some(e) => e,
                        None => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: SymbolExpr::Value(l.clone()), rhs: rhs.clone()})),
                    },
                    _ => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: SymbolExpr::Value(l.clone()), rhs: rhs.clone()})),
                },
                SymbolExpr::Symbol(l) => match rhs {
                    SymbolExpr::Value(r) => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: SymbolExpr::Value(r.clone()), rhs: self.clone()})),
                    SymbolExpr::Symbol(r) => if l.name > r.name {   // sort by name
                        SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: rhs.clone(), rhs: self.clone()}))
                    } else {
                        SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: self.clone(), rhs: rhs.clone()}))
                    },
                    SymbolExpr::Binary(r) => match r.clone().add_opt(self) {
                        Some(e) => e,
                        None => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: self.clone(), rhs: rhs.clone()})),
                    },
                    _ => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: self.clone(), rhs: rhs.clone()})),
                },
                SymbolExpr::Unary(l) => match l.op {
                    UnaryOps::Neg => rhs - &l.expr,
                    _=> SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: self.clone(), rhs: rhs.clone()})),
                },
                SymbolExpr::Binary(l) => match l.add_opt(rhs) {
                    Some(e) => e,
                    None => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Add, lhs: self.clone(), rhs: rhs.clone()}) ),
                }
            }
        }
    }
}

impl Sub for SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        &self - &rhs
    }
}

impl Sub for &SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        if *self == 0.0 {
            -rhs.clone()
        } else if *rhs == 0.0 {
            self.clone()
        } else if *self == *rhs {
            SymbolExpr::Value(Value::Real(0.0))
        } else {
            match rhs {
                SymbolExpr::Value(r) => self + &SymbolExpr::Value(-r),
                SymbolExpr::Unary(r) => match r.op {
                    UnaryOps::Neg => self + &r.expr,
                    _ => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Sub, lhs: self.clone(), rhs: rhs.clone()}) ), 
                },
                _ => match self {
                    SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => match rhs.sub_opt(self) {
                        Some(e) => -e,
                        None => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Sub, lhs: self.clone(), rhs: rhs.clone()}) ),
                    },
                    SymbolExpr::Unary(_) | SymbolExpr::Binary(_) => match self.sub_opt(rhs) {
                        Some(e) => e,
                        None => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Sub, lhs: self.clone(), rhs: rhs.clone()}) ),
                    },
                } ,
            }
        }
    }
}

impl Mul for SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        &self * &rhs
    }
}

impl Mul for &SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        if *self == 0.0 {
            SymbolExpr::Value( Value::Real(0.0))
        } else if *rhs == 0.0 {
            SymbolExpr::Value( Value::Real(0.0))
        } else if *self == 1.0 {
            rhs.clone()
        } else if *rhs == 1.0 {
            self.clone()
        } else if *self == -1.0 {
            rhs.neg()
        } else if *rhs == -1.0 {
            self.neg()
        } else {
            match self {
                SymbolExpr::Value(l) => match rhs {
                    SymbolExpr::Value(r) => SymbolExpr::Value(l * r),
                    SymbolExpr::Binary(r) => match r.mul_opt(self) {
                        Some(e) => e,
                        None => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(l.clone()), rhs: rhs.clone()}) ),
                    },
                    _ => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(l.clone()), rhs: rhs.clone()}) ),
                },
                SymbolExpr::Symbol(l) => match rhs {
                    SymbolExpr::Value(r) => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(r.clone()), rhs: self.clone()}) ),
                    SymbolExpr::Symbol(r) => if l.name > r.name {   // sort by name
                        SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Mul, lhs: rhs.clone(), rhs: self.clone()}))
                    } else {
                        SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Mul, lhs: self.clone(), rhs: rhs.clone()}))
                    },
                    SymbolExpr::Binary(r) => match r.clone().mul_opt(self) {
                        Some(e) => e,
                        None => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: self.clone(), rhs: rhs.clone()})),
                    },
                    _ => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Mul, lhs: self.clone(), rhs: rhs.clone()}) ),
                },
                SymbolExpr::Unary(l) => match l.op {
                    UnaryOps::Neg => -(rhs * &l.expr),
                    UnaryOps::Abs => match rhs {
                        SymbolExpr::Unary(r) => match r.op {
                            UnaryOps::Abs => SymbolExpr::Unary( Arc::new( Unary{ op: UnaryOps::Abs, expr: &l.expr * &r.expr}) ),
                            _=> SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: self.clone(), rhs: rhs.clone()}) ),
                        },
                        _ => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: self.clone(), rhs: rhs.clone()}) ),
                    }
                    _=> SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: self.clone(), rhs: rhs.clone()}) ),
                },
                SymbolExpr::Binary(l) => match l.mul_opt(rhs) {
                    Some(e) => e,
                    None => SymbolExpr::Binary( Arc::new(Binary{ op: BinaryOps::Mul, lhs: self.clone(), rhs: rhs.clone()}) ),
                }
            }
        }
    }
}

impl Div for SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        &self / &rhs
    }
}

impl Div for &SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        if *self == 0.0 {
            SymbolExpr::Value( Value::Real(0.0))
        } else if *rhs == 1.0 {
            self.clone()
        } else if *rhs == -1.0 {
            self.neg()
        } else if *self == *rhs {
            SymbolExpr::Value(Value::Real(1.0))
        } else {
            match self {
                SymbolExpr::Value(l) => match rhs {
                    SymbolExpr::Value(r) => SymbolExpr::Value(l / r),
                    SymbolExpr::Binary(r) => match r.div_opt(self) {
                        Some(e) => e.rcp(),
                        None => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: SymbolExpr::Value(l.clone()), rhs: rhs.clone()}) ),
                    },
                    _ => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: SymbolExpr::Value(l.clone()), rhs: rhs.clone()}) ),
                },
                SymbolExpr::Symbol(_) => match rhs {
                    SymbolExpr::Value(r) => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.clone(), rhs: SymbolExpr::Value(r.clone())}) ),
                    SymbolExpr::Binary(r) => match r.clone().div_opt(self) {
                        Some(e) => e.rcp(),
                        None => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.clone(), rhs: rhs.clone()}) ),
                    },
                    _ => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.clone(), rhs: rhs.clone()}) ),
                },
                SymbolExpr::Unary(l) => match l.op {
                    UnaryOps::Neg => -(&l.expr / rhs),
                    UnaryOps::Abs => match rhs {
                        SymbolExpr::Unary(r) => match r.op {
                            UnaryOps::Abs => SymbolExpr::Unary( Arc::new( Unary{ op: UnaryOps::Abs, expr: &l.expr / &r.expr}) ),
                            _=> SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.clone(), rhs: rhs.clone()}) ),
                        },
                        _ => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.clone(), rhs: rhs.clone()}) ),
                    }
                    _=> SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.clone(), rhs: rhs.clone()}) ),
                },
                SymbolExpr::Binary(l) => match l.div_opt(rhs) {
                    Some(e) => e,
                    None => SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.clone(), rhs: rhs.clone()}) ),
                }
            }
        }
    }
}

impl Neg for SymbolExpr {
    type Output = SymbolExpr;
    fn neg(self) -> SymbolExpr {
        - &self
    }
}

impl Neg for &SymbolExpr {
    type Output = SymbolExpr;
    fn neg(self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value( -l),
            SymbolExpr::Unary(e) => match e.op {
                UnaryOps::Neg => e.expr.clone(),
                _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Neg, expr: self.clone()} )),
            },
            SymbolExpr::Binary(e) => match e.neg_opt() {
                Some(expr) => expr,
                _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Neg, expr: self.clone()} )),
            }
            _ => SymbolExpr::Unary( Arc::new(Unary{ op: UnaryOps::Neg, expr: self.clone()} )),
        }
    }
}

impl PartialEq for SymbolExpr {
    fn eq(&self, rexpr: &Self) -> bool {
        match self {
            SymbolExpr::Symbol(l) => match rexpr {
                SymbolExpr::Symbol(r) => l == r,
                _ => false,
            },
            SymbolExpr::Value(l) => match rexpr {
                SymbolExpr::Value(r) => l == r,
                _ => false,
            },
            SymbolExpr::Unary(l) => match rexpr {
                SymbolExpr::Unary(r) => l == r,
                _ => false,
            },
            SymbolExpr::Binary(l) => match rexpr {
                SymbolExpr::Binary(r) => l == r,
                _ => false,
            },
        }
    }
}

impl PartialEq<f64> for SymbolExpr {
    fn eq(&self, r: &f64) -> bool {
        match self.eval(true) {
            Some(v) => v == *r,
            None => false,
        }
    }
}

impl PartialEq<Complex64> for SymbolExpr {
    fn eq(&self, r: &Complex64) -> bool {
        match self.eval(true) {
            Some(v) => v == *r,
            None => false,
        }
    }
}


// ===============================================================
//  implementations for Symbol
// ===============================================================
impl Symbol {
    pub fn new(expr: &str) -> Self {
        Self { name: expr.to_string()}
    }
    pub fn to_string(&self) -> String {
        self.name.clone()
    }

    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        match maps.get(&self.name) {
            Some(v) => SymbolExpr::Value(v.clone()),
            None =>  SymbolExpr::Symbol(self.clone()),
        }
    }

    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        match maps.get(&self.name) {
            Some(v) => v.clone(),
            None =>  SymbolExpr::Symbol(self.clone()),
        }
    }
}

impl From<&str> for Symbol {
    fn from(v: &str) -> Self {
        Self::new(v)
    }
}

impl PartialEq for Symbol {
    fn eq(&self, r: &Self) -> bool {
        self.name == r.name
    }
}

// ===============================================================
//  implementations for Value
// ===============================================================
impl Value {
    pub fn to_string(&self) -> String {
        match self {
            Value::Real(e) => e.to_string(),
            Value::Complex(e) => if e.re < f64::EPSILON && e.re > -f64::EPSILON {
                if e.im < f64::EPSILON && e.im > -f64::EPSILON {
                    0.to_string()
                } else {
                    String::from(format!("{}i", e.im))
                }
            } else if e.im < f64::EPSILON && e.im > -f64::EPSILON {
                e.re.to_string()
            } else {
                e.to_string()
            },
        }
    }
    pub fn as_real(&self) -> f64 {
        match self {
            Value::Real(e) => *e,
            Value::Complex(e) => (e.re*e.re + e.im*e.im).sqrt(),
        }
    }

    pub fn abs(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.abs()),
            Value::Complex(e) => Value::Real((e.re*e.re + e.im*e.im).sqrt()),
        }
    }
    pub fn sin(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.sin()),
            Value::Complex(e) => Value::Complex(e.sin()),
        }
    }
    pub fn asin(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.asin()),
            Value::Complex(e) => Value::Complex(e.asin()),
        }
    }
    pub fn cos(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.cos()),
            Value::Complex(e) => Value::Complex(e.cos()),
        }
    }
    pub fn acos(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.acos()),
            Value::Complex(e) => Value::Complex(e.acos()),
        }
    }
    pub fn tan(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.tan()),
            Value::Complex(e) => Value::Complex(e.tan()),
        }
    }
    pub fn atan(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.atan()),
            Value::Complex(e) => Value::Complex(e.atan()),
        }
    }
    pub fn exp(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.exp()),
            Value::Complex(e) => Value::Complex(e.exp()),
        }
    }
    pub fn log(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.ln()),
            Value::Complex(e) => Value::Complex(e.ln()),
        }
    }
    pub fn sqrt(self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.sqrt()),
            Value::Complex(e) => Value::Complex(e.sqrt()),
        }
    }
    pub fn pow(self, p: Value) -> Value {
        match self {
            Value::Real(e) => match p {
                Value::Real(r) => if e < 0.0 {
                    Value::Complex(Complex64::from(e).powf(r))
                } else {
                    Value::Real(e.powf(r))
                },
                Value::Complex(r) => Value::Complex(Complex64::from(e).powc(r)),
            },
            Value::Complex(e) => match p {
                Value::Real(r) => Value::Complex(e.powf(r)),
                Value::Complex(r) => Value::Complex(e.powc(r)),
            },
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Real(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Real(v as f64)
    }
}

impl From<Complex64> for Value {
    fn from(v: Complex64) -> Self {
        if v.im < f64::EPSILON && v.im > -f64::EPSILON {
            Value::Real(v.re)
        } else{
            Value::Complex(v)
        }
    }
}


impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        &self + &rhs
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l + r),
                Value::Complex(r) => Value::Complex(l + r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l + r),
                Value::Complex(r) => Value::Complex(l + r),
            },
        }
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        &self - &rhs
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l - r),
                Value::Complex(r) => Value::Complex(l - r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l - r),
                Value::Complex(r) => Value::Complex(l - r),
            },
        }
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        &self * &rhs
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l * r),
                Value::Complex(r) => Value::Complex(l * r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l * r),
                Value::Complex(r) => Value::Complex(l * r),
            },
        }
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        &self / &rhs
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l / r),
                Value::Complex(r) => Value::Complex(l / r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l / r),
                Value::Complex(r) => Value::Complex(l / r),
            },
        }
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        -&self
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        match self {
            Value::Real(v) => Value::Real( -v),
            Value::Complex(v) => Value::Complex( -v),
        }
    }
}


impl PartialEq for Value {
    fn eq(&self, r: &Self) -> bool {
        match self {
            Value::Real(e) => match r {
                Value::Real(rv) => (e - rv).abs() < f64::EPSILON,
                Value::Complex(rv) => {
                    let t = Complex64::from(*e) - rv;
                    return t.re < f64::EPSILON && t.re > -f64::EPSILON && t.im < f64::EPSILON && t.im > -f64::EPSILON;
                },
            },
            Value::Complex(e) => match r {
                Value::Real(rv) => {
                    let t = *e - Complex64::from(rv);
                    return t.re < f64::EPSILON && t.re > -f64::EPSILON && t.im < f64::EPSILON && t.im > -f64::EPSILON;
                },
                Value::Complex(rv) =>{
                    let t = *e - rv;
                    return t.re < f64::EPSILON && t.re > -f64::EPSILON && t.im < f64::EPSILON && t.im > -f64::EPSILON;
                },
            },
        }
    }
}

impl PartialEq<f64> for Value {
    fn eq(&self, r: &f64) -> bool {
        match self {
            Value::Real(e) => (e - r).abs() < f64::EPSILON,
            Value::Complex(e) => {
                let t = *e - Complex64::from(r);
                return t.re < f64::EPSILON && t.re > -f64::EPSILON && t.im < f64::EPSILON && t.im > -f64::EPSILON;
            },
        }
    }
}

impl PartialEq<Complex64> for Value {
    fn eq(&self, r: &Complex64) -> bool {
        match self {
            Value::Real(e) => {
                let t = Complex64::from(*e) - r;
                return t.re < f64::EPSILON && t.re > -f64::EPSILON && t.im < f64::EPSILON && t.im > -f64::EPSILON;
            },
            Value::Complex(e) => {
                let t = *e - r;
                return t.re < f64::EPSILON && t.re > -f64::EPSILON && t.im < f64::EPSILON && t.im > -f64::EPSILON;
            },
        }
    }
}


// ===============================================================
//  implementations for Unary operators
// ===============================================================
impl Unary {
    pub fn new(op: UnaryOps, expr: SymbolExpr) -> Self {
        Self { op: op, expr: expr}
    }
    pub fn to_string(&self) -> String {
        let s = self.expr.to_string();
        match self.op {
            UnaryOps::Abs => String::from(format!("abs({})", s)),
            UnaryOps::Neg => match &self.expr {
                SymbolExpr::Value(e) => String::from(format!("{}", (-e.clone()).to_string())),
                SymbolExpr::Binary(e) => match e.op {
                    BinaryOps::Add | BinaryOps::Sub => String::from(format!("-({})", s)),
                    _ => String::from(format!("-{}", s)),
                },
                _ => String::from(format!("-{}", s)),
            },
            UnaryOps::Sin => String::from(format!("sin({})", s)),
            UnaryOps::Asin => String::from(format!("asin({})", s)),
            UnaryOps::Cos => String::from(format!("cos({})", s)),
            UnaryOps::Acos => String::from(format!("acos({})", s)),
            UnaryOps::Tan => String::from(format!("tan({})", s)),
            UnaryOps::Atan => String::from(format!("atan({})", s)),
            UnaryOps::Exp => String::from(format!("exp({})", s)),
            UnaryOps::Log => String::from(format!("log({})", s)),
        }
    }

    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        let new_expr = Unary{ op: self.op.clone(), expr: self.expr.bind(maps),};
        match new_expr.clone().eval(false) {
            Some(v) => SymbolExpr::Value(v.clone()),
            None => SymbolExpr::Unary( Arc::new(new_expr))
        }
    }

    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        let new_expr = Unary{ op: self.op.clone(), expr: self.expr.subs(maps),};
        match new_expr.clone().eval(false) {
            Some(v) => SymbolExpr::Value(v.clone()),
            None => SymbolExpr::Unary( Arc::new(new_expr))
        }
    }

    pub fn derivative(&self, param: &SymbolExpr) -> SymbolExpr {
        let expr_d = self.expr.derivative(param);
        match self.op {
            UnaryOps::Abs => self.expr.clone() * expr_d / SymbolExpr::Unary( Arc::new( Unary {op: self.op.clone(), expr: self.expr.clone()})),
            UnaryOps::Neg => SymbolExpr::Unary( Arc::new( Unary {op: UnaryOps::Neg, expr: expr_d})),
            UnaryOps::Sin => {
                let lhs = SymbolExpr::Unary( Arc::new( Unary {op: UnaryOps::Cos, expr: self.expr.clone()}));
                lhs * expr_d
            },
            UnaryOps::Asin => {
                let d = SymbolExpr::Value( Value::Real(1.0)) - self.expr.clone() * self.expr.clone();
                let lhs = match d {
                    SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                    _ => SymbolExpr::Binary( Arc::new( Binary {op: BinaryOps::Pow, lhs: d, rhs: SymbolExpr::Value( Value::Real(0.5))} )),
                };
                lhs * expr_d
            },
            UnaryOps::Cos => {
                let lhs = SymbolExpr::Unary( Arc::new( Unary {op: UnaryOps::Sin, expr: self.expr.clone()}));
                -lhs * expr_d
            },
            UnaryOps::Acos => {
                let d = SymbolExpr::Value( Value::Real(1.0)) - self.expr.clone() * self.expr.clone();
                let lhs = match d {
                    SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                    _ => SymbolExpr::Binary( Arc::new( Binary {op: BinaryOps::Pow, lhs: d, rhs: SymbolExpr::Value( Value::Real(0.5))} )),
                };
                -lhs * expr_d
            },
            UnaryOps::Tan =>  {
                let d = SymbolExpr::Unary( Arc::new( Unary {op: UnaryOps::Cos, expr: self.expr.clone()}));
                expr_d / d.clone() / d
            },
            UnaryOps::Atan => {
                let d = SymbolExpr::Value( Value::Real(1.0)) + self.expr.clone() * self.expr.clone();
                expr_d / d
            },
            UnaryOps::Exp => SymbolExpr::Unary( Arc::new( Unary {op: UnaryOps::Exp, expr: self.expr.clone()})) * expr_d,
            UnaryOps::Log => expr_d / self.expr.clone(),
        }       
    }
    pub fn sign(&self) -> f64 {
        match self.op {
            UnaryOps::Abs => 1.0,
            UnaryOps::Neg => -self.expr.sign(),
            _ => match self.expr.eval(true) {
                Some(v) => if v.as_real() > 0.0 {
                    1.0
                } else if v.as_real() < 0.0 {
                    -1.0
                } else {
                    0.0
                },
                None => self.expr.sign(),
            }
        }
    }

    pub fn eval(&self, recurse: bool) -> Option<Value> {
        let val : Value;
        if recurse {
            match self.expr.eval(recurse) {
                Some(v) => val = v,
                None => return None,
            }
        }
        else {
            match &self.expr {
                SymbolExpr::Value(e) => val = e.clone(),
                _ => return None,
            }         
        }
        let ret = match self.op {
            UnaryOps::Abs => val.abs(),
            UnaryOps::Neg => -val,
            UnaryOps::Sin => val.sin(),
            UnaryOps::Asin => val.asin(),
            UnaryOps::Cos => val.cos(),
            UnaryOps::Acos => val.acos(),
            UnaryOps::Tan => val.tan(),
            UnaryOps::Atan => val.atan(),
            UnaryOps::Exp => val.exp(),
            UnaryOps::Log => val.log(),
        };
        match ret {
            Value::Real(_) => Some(ret),
            Value::Complex(c) => if c.im == 0.0 {
                Some(Value::Real(c.re))
            } else {
                Some(ret)
            }
        }
    }

    pub fn symbols(&self) -> HashSet<String> {
        self.expr.symbols()
    }

    pub fn has_symbol(&self, param: &String) -> bool {
        self.expr.has_symbol(param)
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self.op {
            UnaryOps::Neg => match rhs.sub_opt(&self.expr) {
                Some(e) => Some(-e),
                None => None,
            },
            _ => None,
        }       
    }
    // Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self.op {
            UnaryOps::Neg => match rhs.add_opt(&self.expr) {
                Some(e) => Some(-e),
                None => None,
            },
            _ => None,
        }       
    }
}

impl PartialEq for Unary {
    fn eq(&self, r: &Self) -> bool {
        self.op == r.op && self.expr == r.expr
    }
}

// ===============================================================
//  implementations for Binary operators
// ===============================================================
impl Binary {
    pub fn new(op: BinaryOps, lhs: SymbolExpr, rhs: SymbolExpr) -> Self {
        Self { op: op, lhs: lhs, rhs: rhs}
    }

    pub fn to_string(&self) -> String {
        let s_lhs = self.lhs.to_string();
        let s_rhs = self.rhs.to_string();
        let op_lhs = match &self.lhs {
            SymbolExpr::Binary(e) => match e.op {
                BinaryOps::Add | BinaryOps::Sub => true,
                _ => false,
            },
            SymbolExpr::Value(e) => match e {
                Value::Real(v) => *v < 0.0,
                Value::Complex(_) => true,
            },
            _ => false,
        };
        let op_rhs = match &self.rhs {
            SymbolExpr::Binary(e) => match e.op {
                BinaryOps::Add | BinaryOps::Sub => true,
                _ => false,
            },
            SymbolExpr::Value(e) => match e {
                Value::Real(v) => *v < 0.0,
                Value::Complex(_) => true,
            },
            _ => false,
        };

        match self.op {
            BinaryOps::Add => match &self.rhs {
                SymbolExpr::Unary(r) => match r.op {
                    UnaryOps::Neg => if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                        String::from(format!("{} {}", s_lhs, s_rhs))
                    } else {
                        String::from(format!("{} + {}", s_lhs, s_rhs))
                    }
                    _ => String::from(format!("{} + {}", s_lhs, s_rhs)),
                },
                _ => String::from(format!("{} + {}", s_lhs, s_rhs))
            },
            BinaryOps::Sub =>  match &self.rhs {
                SymbolExpr::Unary(r) => match r.op {
                    UnaryOps::Neg => if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                        let st = s_rhs.char_indices().nth(0).unwrap().0;
                        let ed = s_rhs.char_indices().nth(1).unwrap().0;
                        let s_rhs_new: &str = &s_rhs.as_str()[st..ed];
                        String::from(format!("{} + {}", s_lhs, s_rhs_new))
                    } else {
                        if op_rhs {
                            String::from(format!("{} -({})", s_lhs, s_rhs))
                        } else {
                            String::from(format!("{} - {}", s_lhs, s_rhs))
                        }
                    }
                    _ => if op_rhs {
                        String::from(format!("{} -({})", s_lhs, s_rhs))
                    } else {
                        String::from(format!("{} - {}", s_lhs, s_rhs))
                    },
                },
                _ => if op_rhs {
                    String::from(format!("{} -({})", s_lhs, s_rhs))
                } else {
                    String::from(format!("{} - {}", s_lhs, s_rhs))
                },
            },
            BinaryOps::Mul => if op_lhs {
                if op_rhs {
                    String::from(format!("({}) * ({})", s_lhs, s_rhs))
                } else {
                    String::from(format!("({}) * {}", s_lhs, s_rhs))
                }
            } else {
                if op_rhs {
                    String::from(format!("{} * ({})", s_lhs, s_rhs))
                } else {
                    String::from(format!("{} * {}", s_lhs, s_rhs))
                }
            },
            BinaryOps::Div => if op_lhs {
                if op_rhs {
                    String::from(format!("({}) / ({})", s_lhs, s_rhs))
                } else {
                    String::from(format!("({}) / {}", s_lhs, s_rhs))
                }
            } else {
                if op_rhs {
                    String::from(format!("{} / ({})", s_lhs, s_rhs))
                } else {
                    String::from(format!("{} / {}", s_lhs, s_rhs))
                }
            },
            BinaryOps::Pow => match &self.lhs {
                SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => match &self.rhs {
                    SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => String::from(format!("({})**({})", s_lhs, s_rhs)),
                    SymbolExpr::Value(r) => if r.as_real() < 0.0 {
                        String::from(format!("({})**({})", s_lhs, s_rhs))
                    } else {
                        String::from(format!("({})**{}", s_lhs, s_rhs))
                    },
                    _ => String::from(format!("({})**{}", s_lhs, s_rhs)),
                },
                SymbolExpr::Value(l) => if l.as_real() < 0.0 {
                    match &self.rhs {
                        SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => String::from(format!("({})**({})", s_lhs, s_rhs)),
                        _ => String::from(format!("({})**{}", s_lhs, s_rhs)),
                    }
                } else {
                    match &self.rhs {
                        SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => String::from(format!("{}**({})", s_lhs, s_rhs)),
                        _ => String::from(format!("{}**{}", s_lhs, s_rhs)),
                    }
                },
                _ => match &self.rhs {
                    SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => String::from(format!("{}**({})", s_lhs, s_rhs)),                  
                    SymbolExpr::Value(r) => if r.as_real() < 0.0 {
                        String::from(format!("{}**({})", s_lhs, s_rhs))
                    } else {
                        String::from(format!("{}**{}", s_lhs, s_rhs))
                    },
                    _ => String::from(format!("{}**{}", s_lhs, s_rhs)),
                },
            },
        }
    }

    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        let new_expr = Binary{ op: self.op.clone(), lhs: self.lhs.bind(maps), rhs: self.rhs.bind(maps),};
        match new_expr.clone().eval(false) {
            Some(v) => SymbolExpr::Value(v),
            None => match self.op {
                BinaryOps::Add => new_expr.lhs + new_expr.rhs,
                BinaryOps::Sub => new_expr.lhs - new_expr.rhs,
                BinaryOps::Mul => new_expr.lhs * new_expr.rhs,
                BinaryOps::Div => new_expr.lhs / new_expr.rhs,
                BinaryOps::Pow => new_expr.lhs.pow(&new_expr.rhs),
            }
        }
    }

    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        let new_expr = Binary{ op: self.op.clone(), lhs: self.lhs.subs(maps), rhs: self.rhs.subs(maps),};
        match new_expr.clone().eval(false) {
            Some(v) => SymbolExpr::Value(v),
            None => match self.op {
                BinaryOps::Add => new_expr.lhs + new_expr.rhs,
                BinaryOps::Sub => new_expr.lhs - new_expr.rhs,
                BinaryOps::Mul => new_expr.lhs * new_expr.rhs,
                BinaryOps::Div => new_expr.lhs / new_expr.rhs,
                BinaryOps::Pow => new_expr.lhs.pow(&new_expr.rhs),
            }
        }
    }

    pub fn derivative(&self, param: &SymbolExpr) -> SymbolExpr {
        match self.op {
            BinaryOps::Add => self.lhs.derivative(param) + self.rhs.derivative(param),
            BinaryOps::Sub => self.lhs.derivative(param) - self.rhs.derivative(param),
            BinaryOps::Mul => self.lhs.derivative(param) * self.rhs.clone() + self.lhs.clone() * self.rhs.derivative(param),
            BinaryOps::Div => (self.lhs.derivative(param) * self.rhs.clone() - self.lhs.clone() * self.rhs.derivative(param)) / self.rhs.clone() / self.rhs.clone(),
            BinaryOps::Pow => {
                if !self.lhs.has_symbol(&param.to_string()) {
                    if !self.rhs.has_symbol(&param.to_string()) {
                        SymbolExpr::Value( Value::Real(0.0))
                    } else {
                        let rhs = SymbolExpr::Unary( Arc::new( Unary{op: UnaryOps::Log, expr: self.lhs.clone()}));
                        SymbolExpr::Binary( Arc::new( 
                            Binary{
                                op: BinaryOps::Mul,
                                lhs: SymbolExpr::Binary( Arc::new(
                                    Binary{
                                        op: BinaryOps::Pow,
                                        lhs: self.lhs.clone(),
                                        rhs: self.rhs.clone(),
                                    } )),
                                rhs: rhs, }) )
                    }
                } else if !self.rhs.has_symbol(&param.to_string()) {
                    let rhs = self.rhs.clone() - SymbolExpr::Value( Value::Real(1.0));
                    self.rhs.clone() * SymbolExpr::Binary( Arc::new( Binary{op: BinaryOps::Pow, lhs: self.lhs.clone(), rhs: rhs}) )
                } else {
                    let new_expr = SymbolExpr::Unary( Arc::new( 
                        Unary { 
                            op: UnaryOps::Exp,
                            expr: SymbolExpr::Binary( Arc::new(
                                Binary{
                                    op: BinaryOps::Mul,
                                    lhs: SymbolExpr::Unary( Arc::new( 
                                        Unary { 
                                            op: UnaryOps::Log,
                                            expr: self.lhs.clone(),
                                        }
                                    ) ),
                                    rhs: self.rhs.clone(),
                                },
                            ) ),
                        }
                    ) );
                    new_expr.derivative(param)
                }
            },
        }       
    }

    pub fn sign(&self) -> f64 {
        let l = self.lhs.sign();
        let r = self.rhs.sign();
        if l == 0.0 {
            if r == 0.0 {
                0.0
            }
            else {
                match self.op {
                    BinaryOps::Add => r,
                    BinaryOps::Sub => -r,
                    BinaryOps::Mul => 0.0,
                    BinaryOps::Div => 1.0,
                    BinaryOps::Pow => 0.0,
                }
            }
        } else {
            if r == 0.0 {
                match self.op {
                    BinaryOps::Add => l,
                    BinaryOps::Sub => l,
                    BinaryOps::Mul => 0.0,
                    BinaryOps::Div => 0.0,
                    BinaryOps::Pow => 1.0,
                }
            } else {
                match self.op {
                    BinaryOps::Add => l*r,
                    BinaryOps::Sub => -l*r,
                    BinaryOps::Mul => l*r,
                    BinaryOps::Div => l*r,
                    BinaryOps::Pow => if l == 1.0 {
                        1.0
                    } else {
                        match &self.rhs {
                            SymbolExpr::Value(v) => match v {
                                Value::Real(r) => if *r - (*r as u64) as f64 == 0.0 {
                                    if *r as u64 % 2 == 0 {
                                        1.0
                                    } else {
                                        -1.0
                                    }
                                } else {
                                    l
                                },
                                Value::Complex(_) => l,
                            }
                            _ => l,
                        }
                    },
                }
            }
        }
    }

    pub fn eval(&self, recurse: bool) -> Option<Value> {
        let lval : Value;
        let rval : Value;
        if recurse {
            match (self.lhs.eval(true), self.rhs.eval(true)) {
                (Some(left), Some(right)) => {
                    lval = left;
                    rval = right;
                }
                _ => return None,
            }
        }
        else {
            match (&self.lhs, &self.rhs) {
                (SymbolExpr::Value(l), SymbolExpr::Value(r)) => {
                    lval = l.clone();
                    rval = r.clone();
                }
                _ => return None,
            }
        }
        let ret = match self.op {
            BinaryOps::Add => lval + rval,
            BinaryOps::Sub => lval - rval,
            BinaryOps::Mul => lval * rval,
            BinaryOps::Div => lval / rval,
            BinaryOps::Pow => lval.pow(rval),
        };
        match ret {
            Value::Real(_) => Some(ret),
            Value::Complex(c) => if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                Some(Value::Real(c.re))
            } else {
                Some(ret)
            }
        }
    }

    pub fn symbols(&self) -> HashSet<String> {
        let mut symbols = HashSet::<String>::new();
        for s in self.lhs.symbols().union(&self.rhs.symbols()) {
            symbols.insert(s.to_string());
        }
        symbols
    }

    pub fn has_symbol(&self, param: &String) -> bool {
        self.lhs.has_symbol(param) | self.rhs.has_symbol(param)
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let BinaryOps::Add = &self.op {
            if let Some(e) = self.lhs.add_opt(rhs) {
                return match e.add_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: e.clone(), rhs: self.rhs.clone()})) ),
                };
            }
            if let Some(e) = self.rhs.add_opt(rhs) {
                return match self.lhs.add_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: self.lhs.clone(), rhs: e.clone()})) ),
                };
            }
        } else if let BinaryOps::Sub = &self.op {
            if let Some(e) = self.lhs.add_opt(rhs) {
                return match e.add_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: e.clone(), rhs: self.rhs.clone()})) ),
                };
            }
            if let Some(e) = rhs.sub_opt(&self.rhs) {
                return match self.lhs.add_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: self.lhs.clone(), rhs: e.clone()})) ),
                };
            }
        }

        if self.lhs == *rhs {
            match self.op {
                BinaryOps::Add => Some(&(&self.lhs * &SymbolExpr::Value( Value::Real(2.0))) + &self.rhs),
                BinaryOps::Sub => Some(&(&self.lhs * &SymbolExpr::Value( Value::Real(2.0))) - &self.rhs),
                BinaryOps::Mul => match &self.rhs {
                    SymbolExpr::Value(e) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(e.clone() + Value::Real(1.0)), rhs: self.lhs.clone()})) ),
                    _ => None,
                },
                BinaryOps::Div => match &self.rhs {
                    SymbolExpr::Value(e) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: self.lhs.clone(), rhs: SymbolExpr::Value((e.clone() + Value::Real(1.0))/e.clone())})) ),
                    _ => None,
                },
                _ => None,
            }
        } else if self.rhs == *rhs {
            match self.op {
                BinaryOps::Add => Some(&self.lhs + &(&self.rhs * &SymbolExpr::Value( Value::Real(2.0)))),
                BinaryOps::Sub => Some(self.lhs.clone()),
                BinaryOps::Mul => match &self.lhs {
                    SymbolExpr::Value(e) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(e.clone() + Value::Real(1.0)), rhs: self.rhs.clone()})) ),
                    _ => None,
                },
                _ => None,
            }
        } else{
            match rhs {
                SymbolExpr::Value(r) => match (&self.lhs, &self.rhs, &self.op) {
                    (SymbolExpr::Value(l_l), _, BinaryOps::Add | BinaryOps::Sub) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: SymbolExpr::Value(l_l + r), rhs: self.rhs.clone()})) ),
                    (_, SymbolExpr::Value(l_r), BinaryOps::Add) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: SymbolExpr::Value(l_r + r), rhs: self.lhs.clone()})) ),
                    (_, SymbolExpr::Value(l_r), BinaryOps::Sub) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: SymbolExpr::Value(r - l_r), rhs: self.lhs.clone()})) ),
                    (_, _, _) => None,
                },
/*                SymbolExpr::Binary(r) => if r.lhs == self.lhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.lhs * &(&self.rhs + &r.rhs)),
                        (_,_) => None,
                    }
                } else if r.rhs == self.rhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&(&self.lhs + &r.lhs) * &self.rhs),
                        (BinaryOps::Div, BinaryOps::Div) => Some(&(&self.lhs + &r.lhs) / &self.rhs),
                        (_,_) => None,
                    }
                } else if r.rhs == self.lhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.lhs * &(&r.lhs + &self.rhs)),
                        (_,_) => None,
                    }
                } else if r.lhs == self.rhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.rhs * &(&self.lhs + &r.rhs)),
                        (_,_) => None,
                    }
                } else {
                    None
                },*/
                _ => None,
            }
        }
    }

    // Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let BinaryOps::Add = &self.op {
            if let Some(e) = self.lhs.sub_opt(rhs) {
                return match e.add_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: e.clone(), rhs: self.rhs.clone()})) ),
                };
            }
            if let Some(e) = self.rhs.sub_opt(rhs) {
                return match self.lhs.add_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: self.lhs.clone(), rhs: e.clone()})) ),
                };
            }
        } else if let BinaryOps::Sub = &self.op {
            if let Some(e) = self.lhs.sub_opt(rhs) {
                return match e.sub_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: e.clone(), rhs: self.rhs.clone()})) ),
                };
            }
            if let Some(e) = self.rhs.add_opt(rhs) {
                return match self.lhs.sub_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: self.lhs.clone(), rhs: e.clone()})) ),
                };
            }
        }
        if let BinaryOps::Sub = &self.op {
            if let Some(e) = self.rhs.sub_opt(rhs) {
                return match self.lhs.sub_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: self.lhs.clone(), rhs: e.clone()})) ),
                };
            }
        }
        if self.lhs == *rhs {
            match self.op {
                BinaryOps::Add => Some(self.rhs.clone()),
                BinaryOps::Sub => Some(-&self.rhs),
                BinaryOps::Mul => match &self.rhs {
                    SymbolExpr::Value(e) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(e.clone() - Value::Real(1.0)), rhs: self.lhs.clone()})) ),
                    _ => None,
                },
                BinaryOps::Div => match &self.rhs {
                    SymbolExpr::Value(e) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: self.lhs.clone(), rhs: SymbolExpr::Value((Value::Real(1.0) - e.clone())/e.clone())})) ),
                    _ => None,
                },
                _ => None,
            }
        } else if self.rhs == *rhs {
            match self.op {
                BinaryOps::Add => Some(self.lhs.clone()),
                BinaryOps::Sub => Some(&self.lhs - &(&self.rhs * &SymbolExpr::Value( Value::Real(2.0)))),
                BinaryOps::Mul => match &self.lhs {
                    SymbolExpr::Value(e) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(e.clone() - Value::Real(1.0)), rhs: self.rhs.clone()})) ),
                    _ => None,
                },
                _ => None,
            }
        } else{
            match rhs {
                SymbolExpr::Value(r) => match (&self.lhs, &self.rhs, &self.op) {
                    (SymbolExpr::Value(l_l), _, BinaryOps::Add | BinaryOps::Sub) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: SymbolExpr::Value(l_l - r), rhs: self.rhs.clone()})) ),
                    (_, SymbolExpr::Value(l_r), BinaryOps::Add) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: self.lhs.clone(), rhs: SymbolExpr::Value(l_r + r)})) ),
                    (_, SymbolExpr::Value(l_r), BinaryOps::Sub) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: self.lhs.clone(), rhs: SymbolExpr::Value(r - l_r)})) ),
                    (_, _, _) => None,
                },
/*                SymbolExpr::Binary(r) => if r.lhs == self.lhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.lhs * &(&self.rhs - &r.rhs)),
                        (_,_) => None,
                    }
                } else if r.rhs == self.rhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&(&self.lhs - &r.lhs) * &self.rhs),
                        (BinaryOps::Div, BinaryOps::Div) => Some(&(&self.lhs - &r.lhs) / &self.rhs),
                        (_,_) => None,
                    }
                } else if r.rhs == self.lhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.lhs * &(&self.rhs - &r.lhs)),
                        (_,_) => None,
                    }
                } else if r.lhs == self.rhs {
                    match (&self.op, &r.op) {
                        (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.rhs * &(&self.lhs - &r.rhs)),
                        (_,_) => None,
                    }
                } else {
                    None
                },*/
                _ => None,
            }
        }
    }

    // Mul with heuristic optimization
    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if self.rhs == *rhs {
            if let BinaryOps::Div = self.op {
                return Some(self.lhs.clone());
            }
        }
        match rhs {
            SymbolExpr::Value(r) => match (&self.lhs, &self.rhs, &self.op) {
                (SymbolExpr::Value(l_l), _, BinaryOps::Mul | BinaryOps::Div) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: SymbolExpr::Value(l_l * r), rhs: self.rhs.clone()})) ),
                (_, SymbolExpr::Value(l_r), BinaryOps::Mul) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(l_r * r), rhs: self.lhs.clone()})) ),
                (_, SymbolExpr::Value(l_r), BinaryOps::Div) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: SymbolExpr::Value(r / l_r), rhs: self.lhs.clone()})) ),
                (_, _, _) => None,
            },
            SymbolExpr::Binary(r) => if r.rhs == self.lhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Mul, BinaryOps::Div) => Some(&self.rhs * &r.lhs),
                    (BinaryOps::Mul, BinaryOps::Mul) => Some(&(&self.lhs * &r.rhs) * &(&self.rhs * &r.lhs)),
                    (BinaryOps::Div, BinaryOps::Mul) => Some(&(&(&self.lhs * &r.rhs) * &r.lhs) / &self.rhs),
                    (BinaryOps::Div, BinaryOps::Div) => Some(&r.lhs / &self.rhs),
                    (_,_) => None,
                }
            } else if r.lhs == self.rhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Div, BinaryOps::Mul) =>  Some(&self.lhs * &r.rhs),
                    (BinaryOps::Mul, BinaryOps::Mul) =>  Some(&(&self.lhs * &r.rhs) * &(&self.rhs * &r.lhs)),
                    (BinaryOps::Mul, BinaryOps::Div) =>  Some(&(&self.lhs * &(&self.rhs * &r.lhs)) / &r.rhs),
                    (BinaryOps::Div, BinaryOps::Div) =>  Some(&self.lhs / &r.rhs),
                    (_,_) => None,
                }
            } else if r.rhs == self.rhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Mul, BinaryOps::Div) => Some(&self.lhs * &r.lhs),
                    (BinaryOps::Div, BinaryOps::Mul) => Some(&self.lhs * &r.lhs),
                    (BinaryOps::Mul, BinaryOps::Mul) => Some(&(&self.lhs * &r.lhs) * &(&self.rhs * &r.rhs)),
                    (BinaryOps::Div, BinaryOps::Div) => Some(&(&self.lhs * &r.lhs) / &(&self.rhs * &r.rhs)),
                    (_,_) => None,
                }
            } else if r.lhs == self.lhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Mul, BinaryOps::Div) => Some(&(&(&self.lhs * &r.lhs) * &self.rhs) / &r.rhs),
                    (BinaryOps::Div, BinaryOps::Mul) => Some(&(&(&self.lhs * &r.lhs) * &r.rhs) / &self.rhs),
                    (BinaryOps::Mul, BinaryOps::Mul) => Some(&(&self.lhs * &r.lhs) * &(&self.rhs * &r.rhs)),
                    (BinaryOps::Div, BinaryOps::Div) => Some(&(&self.lhs * &r.lhs) / &(&self.rhs * &r.rhs)),
                    (_,_) => None,
                }
            } else {
                None
            },
            _ => None,
        }
    }

    // Div with heuristic optimization
    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if self.lhs == *rhs {
            if let BinaryOps::Mul = self.op {
                return Some(self.rhs.clone());
            } else if let BinaryOps::Div = self.op {
                return Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: SymbolExpr::Value( Value::Real(1.0)), rhs: self.rhs.clone()})) );
            }
        } else if self.rhs == *rhs {
            if let BinaryOps::Mul = self.op {
                return Some(self.lhs.clone());
            }
        }

        match rhs {
            SymbolExpr::Value(r) => match (&self.lhs, &self.rhs, &self.op) {
                (SymbolExpr::Value(l_l), _, BinaryOps::Mul | BinaryOps::Div) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: SymbolExpr::Value(l_l / r), rhs: self.rhs.clone()})) ),
                (_, SymbolExpr::Value(l_r), BinaryOps::Mul) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Div, lhs: self.lhs.clone(), rhs: SymbolExpr::Value(l_r / r)})) ),
                (_, SymbolExpr::Value(l_r), BinaryOps::Div) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Mul, lhs: self.lhs.clone(), rhs: SymbolExpr::Value(r * l_r)})) ),
                (_, _, _) => None,
            },
            SymbolExpr::Binary(r) => if r.rhs == self.lhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.rhs / &r.lhs),
                    (BinaryOps::Mul, BinaryOps::Div) => Some(&(&(&self.lhs * &r.rhs) * &self.rhs) / &r.lhs),
                    (BinaryOps::Div, BinaryOps::Mul) => Some(&SymbolExpr::Value( Value::Real(1.0)) / &(&self.rhs / &r.lhs)),
                    (BinaryOps::Div, BinaryOps::Div) => Some(&(&self.lhs * &r.rhs) / &(&self.rhs * &r.lhs)),
                    (_,_) => None,
                }
            } else if r.lhs == self.rhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.lhs / &r.rhs),
                    (BinaryOps::Mul, BinaryOps::Div) => Some(&self.lhs * &r.rhs),
                    (BinaryOps::Div, BinaryOps::Mul) => Some(&self.lhs / &(&(&self.rhs * &r.lhs) * &r.rhs)),
                    (BinaryOps::Div, BinaryOps::Div) => Some(&(&self.lhs * &r.rhs) / &(&self.rhs * &r.lhs)),
                    (_,_) => None,
                }
            } else if r.rhs == self.rhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.lhs / &r.lhs),
                    (BinaryOps::Mul, BinaryOps::Div) => Some(&(&self.lhs * &(&self.rhs * &r.rhs)) / &r.lhs),
                    (BinaryOps::Div, BinaryOps::Mul) => Some(&self.lhs / &(&r.lhs * &(&self.rhs * &r.rhs))),
                    (BinaryOps::Div, BinaryOps::Div) => Some(&self.lhs / &r.lhs),
                    (_,_) => None,
                }
            } else if r.lhs == self.lhs {
                match (&self.op, &r.op) {
                    (BinaryOps::Mul, BinaryOps::Mul) => Some(&self.rhs / &r.rhs),
                    (BinaryOps::Mul, BinaryOps::Div) => Some(&self.rhs * &r.rhs),
                    (BinaryOps::Div, BinaryOps::Mul) => Some(&SymbolExpr::Value( Value::Real(1.0)) / &(&self.rhs * &r.rhs)),
                    (BinaryOps::Div, BinaryOps::Div) => Some(&r.rhs / &self.rhs),
                    (_,_) => None,
                }
            } else {
                None
            },
            _ => None,
        }
    }

    // optimization of negate
    fn neg_opt(&self) -> Option<SymbolExpr> {
        match self.op {
            BinaryOps::Pow => None,
            _ => {
                let new_lhs = match &self.lhs {
                    SymbolExpr::Value(v) => Some(SymbolExpr::Value(-v)),
                    SymbolExpr::Unary(u) => match u.op {
                        UnaryOps::Neg => Some(u.expr.clone()),
                        _ => None,
                    },
                    SymbolExpr::Binary(b) => b.neg_opt(),
                    _ => None,
                };
                match new_lhs {
                    Some(l) => match self.op {
                        BinaryOps::Add => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: l, rhs: self.rhs.clone()})) ),
                        BinaryOps::Sub => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Add, lhs: l, rhs: self.rhs.clone()})) ),
                        _ => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: l, rhs: self.rhs.clone()})) ),
                    }
                    None => {
                        match self.op {
                            BinaryOps::Mul | BinaryOps::Div => match &self.rhs {
                                SymbolExpr::Value(v) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: self.lhs.clone(), rhs: SymbolExpr::Value(-v)})) ),
                                SymbolExpr::Unary(u) => match u.op {
                                    UnaryOps::Neg => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: self.lhs.clone(), rhs: u.expr.clone()})) ),
                                    _ => None,
                                },
                                SymbolExpr::Binary(b) => match b.neg_opt() {
                                    Some(b_opt) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: self.op.clone(), lhs: self.lhs.clone(), rhs: b_opt})) ),
                                    None => None,
                                }
                                _ => None,
                            },
                            BinaryOps::Add => match &self.rhs {
                                SymbolExpr::Value(v) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: SymbolExpr::Value(-v), rhs: self.lhs.clone()})) ),
                                SymbolExpr::Unary(u) => match u.op {
                                    UnaryOps::Neg => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: u.expr.clone(), rhs: self.lhs.clone()})) ),
                                    _ => None,
                                },
                                SymbolExpr::Binary(b) => match b.neg_opt() {
                                    Some(b_opt) => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: b_opt, rhs: self.lhs.clone()})) ),
                                    None => None,
                                }
                                _ => None,
                            },
                            BinaryOps::Sub => Some(SymbolExpr::Binary( Arc::new( Binary{ op: BinaryOps::Sub, lhs: self.rhs.clone(), rhs: self.lhs.clone()})) ),
                            _ => None,
                        }
                    },
                }
            },
        }
    }
}

impl PartialEq for Binary {
    fn eq(&self, r: &Self) -> bool {
        if self.op != r.op {
            return false;
        }
        match self.op {
            BinaryOps::Add | BinaryOps::Mul => (self.lhs == r.lhs && self.rhs == r.rhs) || (self.lhs == r.rhs && self.rhs == r.lhs),
            _ => self.lhs == r.lhs && self.rhs == r.rhs,
        }
    }
}


