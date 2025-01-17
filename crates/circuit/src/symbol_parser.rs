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

extern crate nom;
extern crate nom_unicode;
use nom::IResult;
use nom::Parser;
use nom::character::complete::{char, multispace0, digit1};
use nom::bytes::complete::tag;
use nom::combinator::{all_consuming, map_res, recognize, opt};
use nom::branch::{alt, permutation};
use nom::sequence::{delimited, pair, tuple};
use nom::multi::{many0, many0_count};
use nom::number::complete::double;

use num_complex::c64;

use std::sync::Arc;
use crate::symbol_expr::{SymbolExpr, BinaryOps, Symbol, Value, Unary, UnaryOps};

#[derive(Clone)]
struct BinaryOpContainer {
    op: BinaryOps,
    expr: SymbolExpr,
}

impl BinaryOpContainer {
    fn accum(self, rhs: BinaryOpContainer) -> BinaryOpContainer {
        match rhs.op {
            BinaryOps::Add => BinaryOpContainer{op: rhs.op, expr: self.expr + rhs.expr,}, 
            BinaryOps::Sub => BinaryOpContainer{op: rhs.op, expr: self.expr - rhs.expr,}, 
            BinaryOps::Mul => BinaryOpContainer{op: rhs.op, expr: self.expr * rhs.expr,}, 
            BinaryOps::Div => BinaryOpContainer{op: rhs.op, expr: self.expr / rhs.expr,}, 
            BinaryOps::Pow => BinaryOpContainer{op: rhs.op, expr: self.expr.pow(&rhs.expr),}, 
        }
    }
}

fn parse_value(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        double,
        |v| -> Result<BinaryOpContainer, &str> {
            Ok(BinaryOpContainer{op: BinaryOps::Add, expr: SymbolExpr::Value( Value::Real(v))})
        }
    )(s)
}

fn parse_imaginary_value(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((
            double,
            char('i'),
        )),
        |(v, _)| -> Result<BinaryOpContainer, &str> {
            Ok(BinaryOpContainer{op: BinaryOps::Add, expr: SymbolExpr::Value( Value::Complex(c64(0.0, v)))})
        }
    )(s)
}

fn alpha1(i: &str) -> IResult<&str, &str> {
    nom_unicode::complete::alpha1(i)
}

fn alphanumeric1(i: &str) -> IResult<&str, &str> {
    nom_unicode::complete::alphanumeric1(i)
}

fn parse_symbol_string(s: &str) -> IResult<&str, &str> {
    recognize(
      pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_"))))
      )
    ).parse(s)
}

fn parse_symbol(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((
            parse_symbol_string,
            opt(
                delimited(
                    char('['),
                    digit1,
                    char(']'),
                ),
            ),
        )),
        |(v, array_idx)| -> Result<BinaryOpContainer, &str> {
            match array_idx {
                Some(i) => {
                    // currently array index is stored as string
                    // if array indexing is required in the future
                    // add indexing in Symbol struct
                    let s = format!("{}[{}]",v,i);
                    return Ok(BinaryOpContainer{op: BinaryOps::Add, expr: SymbolExpr::Symbol( Symbol::new(&s))});
                },
                None => Ok(BinaryOpContainer{op: BinaryOps::Add, expr: SymbolExpr::Symbol( Symbol::new(v))}),
            }
        }
    )(s)
}

fn parse_unary(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((
            alphanumeric1,
            delimited(
                char('('),
                delimited(multispace0, parse_addsub, multispace0),
                char(')'),
            ),
        )),
        |(v, expr)| -> Result<BinaryOpContainer, &str> {
            let op = match v {
                "sin" => UnaryOps::Sin,
                "asin" => UnaryOps::Asin,
                "cos" => UnaryOps::Cos,
                "acos" => UnaryOps::Acos,
                "tan" => UnaryOps::Tan,
                "atan" => UnaryOps::Atan,
                "log" => UnaryOps::Log,
                "exp" => UnaryOps::Exp,
                &_ => return Err("unsupported unary operation found."),
            };
            Ok(BinaryOpContainer{op: BinaryOps::Add, expr: SymbolExpr::Unary( Arc::new(Unary::new(op,expr.expr)))})
        }
    )(s)
}

fn parse_neg(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        tuple((
            char('-'),
            alt((
                parse_imaginary_value,
                parse_value,
                parse_unary,
                parse_symbol,
                delimited(
                    char('('),
                    delimited(multispace0, parse_addsub, multispace0),
                    char(')'),
                ),
            )),
        )),
        |(_, expr)| -> Result<BinaryOpContainer, &str> {
            Ok(BinaryOpContainer{op: BinaryOps::Add, expr: SymbolExpr::Unary( Arc::new(Unary::new(UnaryOps::Neg, expr.expr)))})
        }
    )(s)
}

fn parse_expr(s: &str) -> IResult<&str, BinaryOpContainer> {
    alt((
        parse_imaginary_value,
        parse_value,
        parse_neg,
        parse_unary,
        parse_symbol,
        delimited(
            char('('),
            delimited(multispace0, parse_addsub, multispace0),
            char(')'),
        ),
    ))(s)
}

// parse mul and div and pow
fn parse_muldiv(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        permutation((
            parse_expr,
            many0(
                map_res(
                    tuple((
                        multispace0,
                        alt((tag("**"), tag("*"), tag("/"),)),
                        multispace0,
                        parse_expr,
                    )),
                    |(_, opr, _, mut rhs)| -> Result<BinaryOpContainer, &str> {
                        if opr == "**" {
                            rhs.op = BinaryOps::Pow;
                            Ok(rhs)
                        } else if opr == "*" {
                            rhs.op = BinaryOps::Mul;
                            Ok(rhs)
                        } else {
                            rhs.op = BinaryOps::Div;
                            Ok(rhs)
                        }
                    }
                )
            ),
        )),
        |(lhs, rvec)| -> Result<BinaryOpContainer, &str> {
            Ok(rvec.iter().fold(lhs, |acc, x| { acc.accum(x.clone())}))
        }
    )(s)
}

// parse add and sub
fn parse_addsub(s: &str) -> IResult<&str, BinaryOpContainer> {
    map_res(
        permutation((
            parse_muldiv,
            many0(
                map_res(
                    tuple((
                        multispace0,
                        alt((char('+'), char('-'),)),
                        multispace0,
                        parse_muldiv,
                    )),
                    |(_, opr, _, mut rhs)| -> Result<BinaryOpContainer, &str> {
                        if opr == '+' {
                            rhs.op = BinaryOps::Add;
                            Ok(rhs)
                        } else {
                            rhs.op = BinaryOps::Sub;
                            Ok(rhs)
                        }
                    }
                )
            ),
        )),
        |(lhs, rvec)| -> Result<BinaryOpContainer, &str> {
            Ok(rvec.iter().fold(lhs, |acc, x| { acc.accum(x.clone())}))
        }
    )(s)
}

pub fn parse_expression(s: &str) -> SymbolExpr {
    let mut parser = all_consuming(parse_addsub);
    parser(s).unwrap().1.expr
}


