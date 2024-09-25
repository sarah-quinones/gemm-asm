use std::{env, fmt::Write, fs, path::Path};

#[derive(Copy, Clone)]
pub enum Dtype {
    F64,
    F32,
}

impl std::fmt::Debug for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::F64 => f.write_str("float64"),
            Dtype::F32 => f.write_str("float32"),
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::F64 => f.write_str("d"),
            Dtype::F32 => f.write_str("s"),
        }
    }
}

use x86_64::{RealAvx, RealAvx512};
pub mod x86_64 {
    use std::fmt::Write;

    use crate::Dtype;

    pub struct RealAvx(pub String, pub Dtype);
    pub struct RealAvx512(pub String, pub Dtype);

    impl KernelCtx for RealAvx {
        fn has_bitmask(&self) -> bool {
            false
        }
        fn writeln(&mut self, code: &str) {
            self.0 += code;
            self.0 += "\n";
        }
        fn n_regs(&self) -> usize {
            16
        }
        fn reg_size(&self) -> i32 {
            32
        }
        fn vload(&mut self, xmm: usize, mem: Mem) {
            let dtype = self.1;
            writeln!(self.0, "vmovup{dtype} ymm{xmm}, {mem}").unwrap();
        }
        fn vstore(&mut self, mem: Mem, xmm: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmovup{dtype} {mem}, ymm{xmm}").unwrap();
        }
        fn vbroadcast(&mut self, xmm: usize, mem: Mem) {
            let dtype = self.1;
            writeln!(self.0, "vbroadcasts{dtype} ymm{xmm}, {mem}").unwrap();
        }
        fn vfmadd(&mut self, acc: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vfmadd231p{dtype} ymm{acc}, ymm{lhs}, ymm{rhs}").unwrap();
        }
        fn vmul(&mut self, dst: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmulp{dtype} ymm{dst}, ymm{lhs}, ymm{rhs}").unwrap();
        }
        fn vload_mask(&mut self, mask: usize, mem: Mem) {
            let dtype = self.1;
            writeln!(self.0, "vmovup{dtype} ymm{mask}, {mem}").unwrap();
        }
        fn vand_mask(&mut self, dst: usize, src: usize) {
            let dtype = self.1;
            writeln!(self.0, "vandp{dtype} ymm{dst}, ymm{dst}, ymm{src}").unwrap();
        }
        fn vmaskload(&mut self, mask: usize, xmm: usize, mem: Mem) {
            let dtype = self.1;
            writeln!(self.0, "vmaskmovp{dtype} ymm{xmm}, ymm{mask}, {mem}").unwrap();
        }
        fn vmaskstore(&mut self, mask: usize, mem: Mem, xmm: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmaskmovp{dtype} {mem}, ymm{mask}, ymm{xmm}").unwrap();
        }
        fn vzero(&mut self, xmm: usize) {
            let dtype = self.1;
            writeln!(self.0, "vxorp{dtype} ymm{xmm}, ymm{xmm}, ymm{xmm}").unwrap();
        }
        fn vmov(&mut self, dst: usize, src: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmovap{dtype} ymm{dst}, ymm{src}").unwrap();
        }
        fn loop_nonempty(
            &mut self,
            label: usize,
            counter: Reg,
            body: &mut dyn FnMut(&mut dyn KernelCtx),
        ) {
            self.writeln(&format!("2{label}1:"));
            let this = self;
            body(this);
            this.writeln(&format!("dec {counter}"));
            this.writeln(&format!("jnz 2{label}1b"));
        }
        fn branch_bit(
            &mut self,
            label: usize,
            bit: u32,
            reg: Reg,
            if_true: &mut dyn FnMut(&mut dyn KernelCtx),
            if_false: &mut dyn FnMut(&mut dyn KernelCtx),
        ) {
            let this = self;
            this.writeln(&format!("bt {reg}, {bit}"));
            this.writeln(&format!("jnc 2{label}0f"));
            if_true(this);
            this.writeln(&format!("jmp 2{label}1f"));
            this.writeln(&format!("2{label}0:"));
            if_false(this);
            this.writeln(&format!("2{label}1:"));
        }
    }

    impl KernelCtx for RealAvx512 {
        fn has_bitmask(&self) -> bool {
            true
        }
        fn writeln(&mut self, code: &str) {
            self.0 += code;
            self.0 += "\n";
        }
        fn n_regs(&self) -> usize {
            32
        }
        fn reg_size(&self) -> i32 {
            64
        }
        fn vload(&mut self, xmm: usize, mem: Mem) {
            let dtype = self.1;
            writeln!(self.0, "vmovup{dtype} zmm{xmm}, {mem}").unwrap();
        }
        fn vstore(&mut self, mem: Mem, xmm: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmovup{dtype} {mem}, zmm{xmm}").unwrap();
        }
        fn vbroadcast(&mut self, xmm: usize, mem: Mem) {
            let dtype = self.1;
            writeln!(self.0, "vbroadcasts{dtype} zmm{xmm}, {mem}").unwrap();
        }
        fn vfmadd(&mut self, acc: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vfmadd231p{dtype} zmm{acc}, zmm{lhs}, zmm{rhs}").unwrap();
        }
        fn vmul(&mut self, dst: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmulp{dtype} zmm{dst}, zmm{lhs}, zmm{rhs}").unwrap();
        }
        fn vload_mask(&mut self, mask: usize, mem: Mem) {
            let dtype = match self.1 {
                Dtype::F64 => "b",
                Dtype::F32 => "w",
            };
            writeln!(self.0, "kmov{dtype} k{mask}, {mem}").unwrap();
        }
        fn vand_mask(&mut self, dst: usize, src: usize) {
            let dtype = match self.1 {
                Dtype::F64 => "b",
                Dtype::F32 => "w",
            };
            writeln!(self.0, "kand{dtype} k{dst}, k{dst}, k{src}").unwrap();
        }
        fn vmaskload(&mut self, mask: usize, xmm: usize, mem: Mem) {
            let dtype = self.1;
            writeln!(
                self.0,
                "vmovup{dtype} zmm{xmm} {{{{k{mask}}}}}{{{{z}}}}, {mem}"
            )
            .unwrap();
        }
        fn vmaskstore(&mut self, mask: usize, mem: Mem, xmm: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmovup{dtype} {mem} {{{{k{mask}}}}}, zmm{xmm}").unwrap();
        }
        fn vzero(&mut self, xmm: usize) {
            let dtype = self.1;
            writeln!(self.0, "vxorp{dtype} zmm{xmm}, zmm{xmm}, zmm{xmm}").unwrap();
        }
        fn vmov(&mut self, dst: usize, src: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmovap{dtype} zmm{dst}, zmm{src}").unwrap();
        }
        fn loop_nonempty(
            &mut self,
            label: usize,
            counter: Reg,
            body: &mut dyn FnMut(&mut dyn KernelCtx),
        ) {
            self.writeln(&format!("2{label}1:"));
            let this = self;
            body(this);
            this.writeln(&format!("dec {counter}"));
            this.writeln(&format!("jnz 2{label}1b"));
        }
        fn branch_bit(
            &mut self,
            label: usize,
            bit: u32,
            reg: Reg,
            if_true: &mut dyn FnMut(&mut dyn KernelCtx),
            if_false: &mut dyn FnMut(&mut dyn KernelCtx),
        ) {
            let this = self;
            this.writeln(&format!("bt {reg}, {bit}"));
            this.writeln(&format!("jnc 2{label}0f"));
            if_true(this);
            this.writeln(&format!("jmp 2{label}1f"));
            this.writeln(&format!("2{label}0:"));
            if_false(this);
            this.writeln(&format!("2{label}1:"));
        }
    }

    #[derive(Copy, Clone, Debug, Default)]
    pub enum Reg {
        #[default]
        Rax,
        Rbx,
        Rcx,
        Rdx,
        Rsi,
        Rdi,
        Rbp,
        R8,
        R9,
        R10,
        R11,
        R12,
        R13,
        R14,
        R15,
    }

    impl std::fmt::Display for Reg {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(match self {
                Reg::Rax => "rax",
                Reg::Rbx => "rbx",
                Reg::Rcx => "rcx",
                Reg::Rdx => "rdx",
                Reg::Rsi => "rsi",
                Reg::Rdi => "rdi",
                Reg::Rbp => "rbp",
                Reg::R8 => "r8",
                Reg::R9 => "r9",
                Reg::R10 => "r10",
                Reg::R11 => "r11",
                Reg::R12 => "r12",
                Reg::R13 => "r13",
                Reg::R14 => "r14",
                Reg::R15 => "r15",
            })
        }
    }

    #[derive(Copy, Clone, Debug, Default)]
    pub struct Mem {
        pub addr: Reg,
        pub index: Option<Reg>,
        pub scale: u32,
        pub offset: i32,
    }

    impl std::fmt::Display for Mem {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let Self {
                addr,
                index,
                scale,
                offset,
            } = self;
            let sign = if *offset >= 0 { "+" } else { "-" };
            let offset = offset.unsigned_abs();
            if let Some(index) = index {
                write!(f, "[{addr} + {scale} * {index} {sign} {offset}]")
            } else {
                write!(f, "[{addr} {sign} {offset}]")
            }
        }
    }

    pub trait KernelCtx {
        fn has_bitmask(&self) -> bool;
        fn writeln(&mut self, code: &str);

        fn n_regs(&self) -> usize;
        fn reg_size(&self) -> i32;

        fn push(&mut self, reg: Reg) {
            self.writeln(&format!("push {reg}"));
        }
        fn pop(&mut self, reg: Reg) {
            self.writeln(&format!("pop {reg}"));
        }
        fn lea(&mut self, reg: Reg, addr: Mem) {
            self.writeln(&format!("lea {reg}, {addr}"));
        }
        fn mov(&mut self, dst: Reg, src: Reg) {
            self.writeln(&format!("mov {dst}, {src}"));
        }
        fn add(&mut self, dst: Reg, src: Reg) {
            self.writeln(&format!("add {dst}, {src}"));
        }
        fn sub(&mut self, dst: Reg, src: Reg) {
            self.writeln(&format!("sub {dst}, {src}"));
        }
        fn vzeroupper(&mut self) {
            self.writeln("vzeroupper");
        }
        fn ret(&mut self) {
            self.writeln("ret");
        }

        fn loop_(&mut self, label: usize, counter: Reg, body: &mut dyn FnMut(&mut dyn KernelCtx)) {
            self.writeln(&format!("test {counter}, {counter}"));
            self.writeln(&format!("jz 3{label}1f"));
            self.loop_nonempty(label, counter, body);
            self.writeln(&format!("3{label}1:"));
        }

        fn loop_nonempty(
            &mut self,
            label: usize,
            counter: Reg,
            body: &mut dyn FnMut(&mut dyn KernelCtx),
        );

        fn branch_bit(
            &mut self,
            label: usize,
            bit: u32,
            reg: Reg,
            if_true: &mut dyn FnMut(&mut dyn KernelCtx),
            if_false: &mut dyn FnMut(&mut dyn KernelCtx),
        );

        fn vload(&mut self, xmm: usize, mem: Mem);
        fn vstore(&mut self, mem: Mem, xmm: usize);
        fn vbroadcast(&mut self, xmm: usize, mem: Mem);
        fn vfmadd(&mut self, acc: usize, lhs: usize, rhs: usize);
        fn vmul(&mut self, dst: usize, lhs: usize, rhs: usize);

        fn vload_mask(&mut self, mask: usize, mem: Mem);
        fn vand_mask(&mut self, dst: usize, src: usize);
        fn vmaskload(&mut self, mask: usize, xmm: usize, mem: Mem);
        fn vmaskstore(&mut self, mask: usize, mem: Mem, xmm: usize);

        fn vzero(&mut self, xmm: usize);
        fn vmov(&mut self, dst: usize, src: usize);
    }

    const DEPTH: Reg = Reg::Rax;
    const LHS_CS: Reg = Reg::Rdi;
    const RHS_RS: Reg = Reg::Rcx;
    const RHS_CS: Reg = Reg::Rdx;
    const DST_CS: Reg = Reg::Rsi;
    const TOP_MASK_PTR: Reg = Reg::Rbx;
    const BOT_MASK_PTR: Reg = Reg::Rbp;
    const ALPHA_PTR: Reg = Reg::R8;

    const LHS_PTR: Reg = Reg::R9;
    const RHS_PTR: Reg = Reg::R10;
    const DST_PTR: Reg = Reg::R11;
    const FLAGS: Reg = Reg::R12;

    pub fn microkernel(ctx: &mut dyn KernelCtx, m: usize, n: usize) {
        ctx.branch_bit(
            20,
            1,
            FLAGS,
            &mut |ctx| {
                ctx.branch_bit(
                    21,
                    2,
                    FLAGS,
                    &mut |ctx| microkernel_impl(ctx, m, n, true, true),
                    &mut |ctx| microkernel_impl(ctx, m, n, true, false),
                )
            },
            &mut |ctx| {
                ctx.branch_bit(
                    21,
                    2,
                    FLAGS,
                    &mut |ctx| microkernel_impl(ctx, m, n, false, true),
                    &mut |ctx| microkernel_impl(ctx, m, n, false, false),
                )
            },
        );
        ctx.ret();
    }

    pub fn microkernel_impl(
        ctx: &mut dyn KernelCtx,
        m: usize,
        n: usize,
        use_top_mask: bool,
        use_bot_mask: bool,
    ) {
        fn setup_cs(ctx: &mut dyn KernelCtx, tmp: &[Reg], cs: Reg) {
            if tmp.len() > 1 {
                ctx.lea(
                    tmp[tmp.len() - 1],
                    Mem {
                        addr: cs,
                        index: Some(cs),
                        scale: 2,
                        offset: 0,
                    },
                );
                for i in 1..tmp.len() {
                    ctx.lea(
                        tmp[i],
                        Mem {
                            addr: tmp[i - 1],
                            index: Some(tmp[tmp.len() - 1]),
                            scale: 1,
                            offset: 0,
                        },
                    );
                }
            }
        }

        assert!(m <= 2);

        let tmp0 = Reg::R12;
        let tmp1 = Reg::R13;

        ctx.push(tmp0);
        ctx.push(tmp1);

        let tmp = [RHS_PTR, tmp0, tmp1, DST_PTR];
        let tmp = &tmp[..1 + (n - 1) / 3];

        ctx.vzero(0);
        for i in 1..m * n {
            ctx.vmov(i, 0);
        }

        let top_mask = if ctx.has_bitmask() { 1 } else { m * n + 3 };
        let bot_mask = if ctx.has_bitmask() {
            2
        } else if !use_top_mask {
            m * n + 3
        } else if m * n + 4 < ctx.n_regs() {
            m * n + 4
        } else {
            m * n + 1
        };

        if use_bot_mask {
            ctx.vload_mask(
                bot_mask,
                Mem {
                    addr: BOT_MASK_PTR,
                    ..Default::default()
                },
            );
        }

        if use_top_mask {
            ctx.vload_mask(
                top_mask,
                Mem {
                    addr: TOP_MASK_PTR,
                    ..Default::default()
                },
            );
        }

        if use_top_mask && use_bot_mask && m == 1 {
            ctx.vand_mask(top_mask, bot_mask);
        }

        ctx.push(DEPTH);
        ctx.push(LHS_PTR);
        ctx.push(RHS_PTR);
        {
            if n >= 9 {
                ctx.push(DST_PTR);
            }

            setup_cs(ctx, tmp, RHS_CS);

            ctx.loop_(0, DEPTH, &mut |ctx| {
                let rhs = m * n + m;

                for i in 0..m {
                    let lhs = i + m * n;
                    let offset = ctx.reg_size() * i as i32;

                    let mem = Mem {
                        addr: LHS_PTR,
                        index: None,
                        scale: 0,
                        offset,
                    };

                    if use_top_mask && i == 0 {
                        ctx.vmaskload(top_mask, lhs, mem);
                    } else if use_bot_mask && i + 1 == m {
                        if !ctx.has_bitmask() && bot_mask == lhs {
                            ctx.vload_mask(
                                bot_mask,
                                Mem {
                                    addr: BOT_MASK_PTR,
                                    ..Default::default()
                                },
                            );
                        }
                        ctx.vmaskload(bot_mask, lhs, mem);
                    } else {
                        ctx.vload(lhs, mem);
                    }
                }

                for j in 0..n {
                    let addr = tmp[j / 3];
                    let (index, scale) = match j % 3 {
                        0 => (None, 0),
                        1 => (Some(RHS_CS), 1),
                        2 => (Some(RHS_CS), 2),
                        _ => unreachable!(),
                    };

                    let mem = Mem {
                        addr,
                        index,
                        scale,
                        offset: 0,
                    };
                    ctx.vbroadcast(rhs, mem);

                    for i in 0..m {
                        let lhs = i + m * n;
                        ctx.vfmadd(i + m * j, lhs, rhs);
                    }
                }

                ctx.add(LHS_PTR, LHS_CS);
                for tmp in tmp {
                    ctx.add(*tmp, RHS_RS);
                }
            });

            if n >= 9 {
                ctx.pop(DST_PTR);
            }
        }

        let tmp = [DST_PTR, tmp0, tmp1, RHS_PTR];
        let tmp = &tmp[..1 + (n - 1) / 3];
        setup_cs(ctx, tmp, DST_CS);

        let alpha = m * n;
        let dst = m * n + 2;

        if use_bot_mask && !ctx.has_bitmask() && bot_mask == m * n + 1 {
            ctx.vload_mask(
                bot_mask,
                Mem {
                    addr: BOT_MASK_PTR,
                    ..Default::default()
                },
            );
        }

        ctx.vbroadcast(
            alpha,
            Mem {
                addr: ALPHA_PTR,
                index: None,
                scale: 0,
                offset: 0,
            },
        );

        let update = |ctx: &mut dyn KernelCtx, load: bool| {
            for j in 0..n {
                let addr = tmp[j / 3];
                let (index, scale) = match j % 3 {
                    0 => (None, 0),
                    1 => (Some(DST_CS), 1),
                    2 => (Some(DST_CS), 2),
                    _ => unreachable!(),
                };

                for i in 0..m {
                    let offset = ctx.reg_size() * i as i32;

                    let mem = Mem {
                        addr,
                        index,
                        scale,
                        offset,
                    };

                    if load {
                        if use_top_mask && i == 0 {
                            ctx.vmaskload(top_mask, dst, mem);
                        } else if use_bot_mask && i + 1 == m {
                            ctx.vmaskload(bot_mask, dst, mem);
                        } else {
                            ctx.vload(dst, mem);
                        }
                        ctx.vfmadd(dst, i + m * j, alpha);
                    } else {
                        ctx.vmul(dst, i + m * j, alpha);
                    }

                    if use_top_mask && i == 0 {
                        ctx.vmaskstore(top_mask, mem, dst);
                    } else if use_bot_mask && i + 1 == m {
                        ctx.vmaskstore(bot_mask, mem, dst);
                    } else {
                        ctx.vstore(mem, dst);
                    }
                }
            }
            // ctx.writeln("ud2");
        };

        ctx.branch_bit(
            1, //
            0,
            FLAGS,
            &mut |ctx| update(ctx, true),
            &mut |ctx| update(ctx, false),
        );

        ctx.pop(RHS_PTR);
        ctx.pop(LHS_PTR);
        ctx.pop(DEPTH);
        ctx.pop(tmp1);
        ctx.pop(tmp0);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut codegen = String::new();
    let mut bindings = String::new();

    for dtype in [Dtype::F32, Dtype::F64] {
        // avx
        {
            let max_n = 6;
            let max_m = 2;

            writeln!(bindings, "extern {{")?;
            let mut names = vec![];
            for j in 0..max_n {
                let n = j + 1;
                for i in 0..max_m {
                    let m = i + 1;

                    let name = format!("libfaer_gemm_{dtype:?}_avx_{m}x{n}");
                    writeln!(codegen, ".globl {name}")?;
                    writeln!(codegen, "{name}:")?;

                    let mut ctx = RealAvx(String::new(), Dtype::F64);
                    x86_64::microkernel(&mut ctx, m, n);
                    writeln!(codegen, "{}", ctx.0)?;
                    writeln!(bindings, "pub fn {name}();")?;
                    names.push(name);
                }
            }

            writeln!(bindings, "}}")?;
            writeln!(
                bindings,
                "pub const UKR_AVX_{dtype:?}: &[[unsafe extern fn(); {max_m}]; {max_n}] = &"
            )?;
            writeln!(bindings, "[")?;
            for n in 0..max_n {
                writeln!(bindings, "[")?;
                for m in 0..max_m {
                    writeln!(bindings, "{},", names[m + max_m * n])?;
                }
                writeln!(bindings, "],")?;
            }
            writeln!(bindings, "];")?;
        }

        // avx512
        {
            let max_n = 12;
            let max_m = 2;

            writeln!(bindings, "extern {{")?;
            let mut names = vec![];
            for j in 0..max_n {
                let n = j + 1;
                for i in 0..max_m {
                    let m = i + 1;

                    let name = format!("libfaer_gemm_{dtype:?}_avx512_{m}x{n}");
                    writeln!(codegen, ".globl {name}")?;
                    writeln!(codegen, "{name}:")?;

                    let mut ctx = RealAvx512(String::new(), Dtype::F64);
                    x86_64::microkernel(&mut ctx, m, n);
                    writeln!(codegen, "{}", ctx.0)?;
                    writeln!(bindings, "pub fn {name}();")?;
                    names.push(name);
                }
            }

            writeln!(bindings, "}}")?;
            writeln!(
                bindings,
                "pub const UKR_AVX512_{dtype:?}: &[[unsafe extern fn(); {max_m}]; {max_n}] = &"
            )?;
            writeln!(bindings, "[")?;
            for n in 0..max_n {
                writeln!(bindings, "[")?;
                for m in 0..max_m {
                    writeln!(bindings, "{},", names[m + max_m * n])?;
                }
                writeln!(bindings, "],")?;
            }
            writeln!(bindings, "];")?;
        }
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("codegen.s");
    fs::write(&dest_path, &codegen)?;

    let dest_path = Path::new(&out_dir).join("bindings.rs");
    fs::write(&dest_path, &bindings)?;

    println!("cargo::rerun-if-changed=build.rs");
    Ok(())
}
