use std::{env, fmt::Write, fs, path::Path};

const UNROLL4: bool = true;

#[derive(Copy, Clone)]
pub enum DType {
    F64,
    F32,
}

#[derive(Copy, Clone)]
pub enum Domain {
    Real,
    Cplx,
}

impl DType {
    fn size(&self) -> i32 {
        match self {
            DType::F64 => 8,
            DType::F32 => 4,
        }
    }
}

impl std::fmt::Debug for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Real => f.write_str("real"),
            Domain::Cplx => f.write_str("cplx"),
        }
    }
}

impl std::fmt::Debug for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F64 => f.write_str("float64"),
            DType::F32 => f.write_str("float32"),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F64 => f.write_str("d"),
            DType::F32 => f.write_str("s"),
        }
    }
}

pub mod x86_64 {
    use super::*;

    pub const VERTICAL_RHS_RS: Reg = Reg::Rcx;
    pub const HORIZONTAL_DST_RS: Reg = Reg::Rcx;

    pub const DEPTH: Reg = Reg::Rax;
    pub const LHS_CS: Reg = Reg::Rdi;
    pub const RHS_CS: Reg = Reg::Rdx;
    pub const DST_CS: Reg = Reg::Rsi;
    pub const TOP_MASK_PTR: Reg = Reg::Rbx;
    pub const BOT_MASK_PTR: Reg = Reg::Rbp;
    pub const ALPHA_PTR: Reg = Reg::R8;

    pub const LHS_PTR: Reg = Reg::R9;
    pub const RHS_PTR: Reg = Reg::R10;
    pub const DST_PTR: Reg = Reg::R11;
    pub const FLAGS: Reg = Reg::R12;

    pub struct RealAvx(pub String, pub DType, pub Domain);
    pub struct RealAvx512(pub String, pub DType, pub Domain);

    impl KernelCtx for RealAvx {
        fn vreg(&self) -> &'static str {
            "ymm"
        }
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
        fn unit_size(&self) -> i32 {
            self.1.size()
        }
        fn is_cplx(&self) -> bool {
            matches!(self.2, Domain::Cplx)
        }
        fn vswap(&mut self, xmm: usize) {
            let dtype = self.1;
            let bits = match dtype {
                DType::F64 => 0b01010101,
                DType::F32 => 0b10110001,
            };
            writeln!(self.0, "vpermilp{dtype} ymm{xmm}, ymm{xmm}, {bits}").unwrap();
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
            let instr = match self.2 {
                Domain::Real => "add",
                Domain::Cplx => "subadd",
            };
            writeln!(self.0, "vfm{instr}231p{dtype} ymm{acc}, ymm{lhs}, ymm{rhs}").unwrap();
        }
        fn vfmadd_conj(&mut self, acc: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vfmaddsub231p{dtype} ymm{acc}, ymm{lhs}, ymm{rhs}").unwrap();
        }
        fn vmul(&mut self, dst: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmulp{dtype} ymm{dst}, ymm{lhs}, ymm{rhs}").unwrap();
        }
        fn vadd(&mut self, dst: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vaddp{dtype} ymm{dst}, ymm{lhs}, ymm{rhs}").unwrap();
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
        fn vxor(&mut self, dst: usize, src: usize) {
            let dtype = self.1;
            writeln!(self.0, "vxorp{dtype} ymm{dst}, ymm{dst}, ymm{src}").unwrap();
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
            self.writeln(".align 32");
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

        fn vre_im(&mut self, re: usize, im: usize) {
            if self.is_cplx() {
                match self.1 {
                    DType::F64 => {
                        writeln!(self.0, "vunpckhpd ymm{im}, ymm{re}, ymm{re}").unwrap();
                        writeln!(self.0, "vunpcklpd ymm{re}, ymm{re}, ymm{re}").unwrap();
                    }
                    DType::F32 => {
                        writeln!(self.0, "vmovshdup ymm{im}, ymm{re}").unwrap();
                        writeln!(self.0, "vmovsldup ymm{re}, ymm{re}").unwrap();
                    }
                }
            }
        }

        fn vreduce(&mut self, xmm: usize, tmp: usize) {
            let dtype = self.1;

            writeln!(self.0, "vextractf128 xmm{tmp}, ymm{xmm}, 1").unwrap();
            writeln!(self.0, "vaddp{dtype} xmm{xmm}, xmm{xmm}, xmm{tmp}").unwrap();

            if dtype.size() == 8 && self.is_cplx() {
                return;
            }

            writeln!(self.0, "vunpckhpd xmm{tmp}, xmm{xmm}, xmm{xmm}").unwrap();
            if dtype.size() == 4 {
                writeln!(self.0, "vaddp{dtype} xmm{xmm}, xmm{xmm}, xmm{tmp}").unwrap();
                if self.is_cplx() {
                    return;
                }
                writeln!(self.0, "vunpcklps xmm{tmp}, xmm{xmm}, xmm{xmm}").unwrap();
                writeln!(self.0, "vpsrldq xmm{tmp}, xmm{tmp}, 8").unwrap();
            }
            writeln!(self.0, "vadds{dtype} xmm{xmm}, xmm{xmm}, xmm{tmp}").unwrap();
        }
    }

    impl KernelCtx for RealAvx512 {
        fn vreg(&self) -> &'static str {
            "zmm"
        }

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
        fn unit_size(&self) -> i32 {
            self.1.size()
        }
        fn is_cplx(&self) -> bool {
            matches!(self.2, Domain::Cplx)
        }
        fn vswap(&mut self, xmm: usize) {
            let dtype = self.1;
            let bits = match dtype {
                DType::F64 => 0b01010101,
                DType::F32 => 0b10110001,
            };
            writeln!(self.0, "vpermilp{dtype} zmm{xmm}, zmm{xmm}, {bits}").unwrap();
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
            let instr = match self.2 {
                Domain::Real => "add",
                Domain::Cplx => "subadd",
            };
            writeln!(self.0, "vfm{instr}231p{dtype} zmm{acc}, zmm{lhs}, zmm{rhs}").unwrap();
        }
        fn vfmadd_conj(&mut self, acc: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vfmaddsub231p{dtype} zmm{acc}, zmm{lhs}, zmm{rhs}").unwrap();
        }
        fn vmul(&mut self, dst: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vmulp{dtype} zmm{dst}, zmm{lhs}, zmm{rhs}").unwrap();
        }
        fn vadd(&mut self, dst: usize, lhs: usize, rhs: usize) {
            let dtype = self.1;
            writeln!(self.0, "vaddp{dtype} zmm{dst}, zmm{lhs}, zmm{rhs}").unwrap();
        }
        fn vload_mask(&mut self, mask: usize, mem: Mem) {
            let dtype = match self.1 {
                DType::F64 => "b",
                DType::F32 => "w",
            };
            writeln!(self.0, "kmov{dtype} k{mask}, {mem}").unwrap();
        }
        fn vand_mask(&mut self, dst: usize, src: usize) {
            let dtype = match self.1 {
                DType::F64 => "b",
                DType::F32 => "w",
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
        fn vxor(&mut self, dst: usize, src: usize) {
            let dtype = self.1;
            writeln!(self.0, "vxorp{dtype} zmm{dst}, zmm{dst}, zmm{src}").unwrap();
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
            self.writeln(".align 32");
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

        fn vre_im(&mut self, re: usize, im: usize) {
            if self.is_cplx() {
                match self.1 {
                    DType::F64 => {
                        writeln!(self.0, "vunpckhpd zmm{im}, zmm{re}, zmm{re}").unwrap();
                        writeln!(self.0, "vunpcklpd zmm{re}, zmm{re}, zmm{re}").unwrap();
                    }
                    DType::F32 => {
                        writeln!(self.0, "vmovshdup zmm{im}, zmm{re}").unwrap();
                        writeln!(self.0, "vmovsldup zmm{re}, zmm{re}").unwrap();
                    }
                }
            }
        }

        fn vreduce(&mut self, xmm: usize, tmp: usize) {
            let dtype = self.1;
            writeln!(self.0, "vextractf64x4 ymm{tmp}, zmm{xmm}, 1").unwrap();
            writeln!(self.0, "vaddp{dtype} ymm{xmm}, ymm{xmm}, ymm{tmp}").unwrap();

            writeln!(self.0, "vextractf64x2 xmm{tmp}, ymm{xmm}, 1").unwrap();
            writeln!(self.0, "vaddp{dtype} xmm{xmm}, xmm{xmm}, xmm{tmp}").unwrap();

            if dtype.size() == 8 && self.is_cplx() {
                return;
            }

            writeln!(self.0, "vunpckhpd xmm{tmp}, xmm{xmm}, xmm{xmm}").unwrap();
            if dtype.size() == 4 {
                writeln!(self.0, "vaddp{dtype} xmm{xmm}, xmm{xmm}, xmm{tmp}").unwrap();
                if self.is_cplx() {
                    return;
                }
                writeln!(self.0, "vunpcklps xmm{tmp}, xmm{xmm}, xmm{xmm}").unwrap();
                writeln!(self.0, "vpsrldq xmm{tmp}, xmm{tmp}, 8").unwrap();
            }
            writeln!(self.0, "vadds{dtype} xmm{xmm}, xmm{xmm}, xmm{tmp}").unwrap();
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
                if offset == 0 {
                    write!(f, "[{addr} + {scale} * {index}]")
                } else {
                    write!(f, "[{addr} + {scale} * {index} {sign} {offset}]")
                }
            } else {
                if offset == 0 {
                    write!(f, "[{addr}]")
                } else {
                    write!(f, "[{addr} {sign} {offset}]")
                }
            }
        }
    }

    pub trait KernelCtx {
        fn has_bitmask(&self) -> bool;
        fn writeln(&mut self, code: &str);

        fn n_regs(&self) -> usize;
        fn reg_size(&self) -> i32;
        fn unit_size(&self) -> i32;
        fn is_cplx(&self) -> bool;
        fn vreg(&self) -> &'static str;

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
            if UNROLL4 {
                self.writeln(&format!("push {counter}"));
                self.writeln(&format!("shr {counter}, 2"));
                self.writeln(&format!("jz 3{label}1f"));
                {
                    let mut body = |this: &mut dyn KernelCtx| {
                        body(this);
                        body(this);
                        body(this);
                        body(this);
                    };
                    self.loop_nonempty(label, counter, &mut body);
                }
                self.writeln(&format!("3{label}1:"));
                self.writeln(&format!("pop {counter}"));
                self.writeln(&format!("and {counter}, 3"));
                self.writeln(&format!("jz 3{label}1f"));
                self.loop_nonempty(label, counter, body);
                self.writeln(&format!("3{label}1:"));
            } else {
                self.writeln(&format!("test {counter}, {counter}"));
                self.writeln(&format!("jz 3{label}1f"));
                self.loop_nonempty(label, counter, body);
                self.writeln(&format!("3{label}1:"));
            }
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

        fn vswap(&mut self, xmm: usize);
        fn vload(&mut self, xmm: usize, mem: Mem);
        fn vstore(&mut self, mem: Mem, xmm: usize);
        fn vbroadcast(&mut self, xmm: usize, mem: Mem);
        fn vfmadd(&mut self, acc: usize, lhs: usize, rhs: usize);
        fn vfmadd_conj(&mut self, acc: usize, lhs: usize, rhs: usize);
        fn vmul(&mut self, dst: usize, lhs: usize, rhs: usize);
        fn vadd(&mut self, dst: usize, lhs: usize, rhs: usize);

        fn vload_mask(&mut self, mask: usize, mem: Mem);
        fn vand_mask(&mut self, dst: usize, src: usize);
        fn vmaskload(&mut self, mask: usize, xmm: usize, mem: Mem);
        fn vmaskstore(&mut self, mask: usize, mem: Mem, xmm: usize);

        fn vzero(&mut self, xmm: usize);
        fn vxor(&mut self, dst: usize, src: usize);
        fn vmov(&mut self, dst: usize, src: usize);

        fn vre_im(&mut self, re: usize, im: usize);
        fn vreduce(&mut self, xmm: usize, tmp: usize);
    }

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
        ctx.vzeroupper();
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

        let tmp0 = Reg::R13;
        let tmp1 = Reg::R14;

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

        let nbits = ctx.unit_size() * 8;
        let vreg = ctx.vreg();

        ctx.push(DEPTH);
        ctx.push(LHS_PTR);
        ctx.push(RHS_PTR);
        {
            if n >= 9 {
                ctx.push(DST_PTR);
            }

            setup_cs(ctx, tmp, RHS_CS);

            let body = |ctx: &mut dyn KernelCtx, conj: bool| {
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
                ctx.add(LHS_PTR, LHS_CS);

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

                    if !ctx.is_cplx() && (j + 1 == n || j % 3 == 2) {
                        ctx.add(addr, VERTICAL_RHS_RS);
                    }

                    for i in 0..m {
                        let lhs = i + m * n;
                        if conj {
                            ctx.vfmadd_conj(i + m * j, lhs, rhs);
                        } else {
                            ctx.vfmadd(i + m * j, lhs, rhs);
                        }
                    }
                }

                if ctx.is_cplx() {
                    for i in 0..m {
                        let lhs = i + m * n;
                        ctx.vswap(lhs);
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
                            offset: ctx.unit_size(),
                        };
                        ctx.vbroadcast(rhs, mem);
                        if j + 1 == n || j % 3 == 2 {
                            ctx.add(addr, VERTICAL_RHS_RS);
                        }

                        for i in 0..m {
                            let lhs = i + m * n;
                            if conj {
                                ctx.vfmadd_conj(i + m * j, lhs, rhs);
                            } else {
                                ctx.vfmadd(i + m * j, lhs, rhs);
                            }
                        }
                    }
                }
            };

            if ctx.is_cplx() {
                let sign = m * n + m;
                ctx.branch_bit(
                    1,
                    3,
                    FLAGS,
                    &mut |ctx| {
                        ctx.loop_(0, DEPTH, &mut |ctx| {
                            body(ctx, true);

                            ctx.writeln(&format!(
                                "vmovupd {vreg}{sign}, [rip + LIBFAER_GEMM_COMPLEX{nbits}_MASK_REAL]"
                            ));
                        })
                    },
                    &mut |ctx| {
                        ctx.loop_(0, DEPTH, &mut |ctx| {
                            body(ctx, false);
                            ctx.vzero(sign);
                        })
                    },
                );

                ctx.branch_bit(
                    1,
                    4,
                    FLAGS,
                    &mut |ctx| {
                        ctx.writeln(&format!(
                            "vorpd {vreg}{sign}, {vreg}{sign}, [rip + LIBFAER_GEMM_COMPLEX{nbits}_MASK_IMAG]"
                        ));
                    },
                    &mut |_| {},
                );

                for i in 0..m * n {
                    ctx.vxor(i, sign);
                }
            } else {
                ctx.loop_(0, DEPTH, &mut |ctx| body(ctx, false));
            }

            if n >= 9 {
                ctx.pop(DST_PTR);
            }
        }

        let tmp = [DST_PTR, tmp0, tmp1, RHS_PTR];
        let tmp = &tmp[..1 + (n - 1) / 3];
        setup_cs(ctx, tmp, DST_CS);

        let alpha = m * n;
        let alpha_imag = m * n + 1;
        let dst = m * n + 2;
        ctx.vbroadcast(
            alpha,
            Mem {
                addr: ALPHA_PTR,
                index: None,
                scale: 0,
                offset: 0,
            },
        );

        if ctx.is_cplx() {
            ctx.vbroadcast(
                alpha_imag,
                Mem {
                    addr: ALPHA_PTR,
                    index: None,
                    scale: 0,
                    offset: ctx.unit_size(),
                },
            );

            for i in 0..m * n {
                ctx.vzero(dst);
                ctx.vfmadd_conj(dst, i, alpha);
                ctx.vswap(i);
                ctx.vfmadd_conj(dst, i, alpha_imag);
                ctx.vmov(i, dst);
            }

            let sign = dst;
            ctx.writeln(&format!(
                "vmovupd {vreg}{sign}, [rip + LIBFAER_GEMM_COMPLEX{nbits}_MASK_REAL]"
            ));
            for i in 0..m * n {
                ctx.vxor(i, sign);
            }
        }

        if use_bot_mask && !ctx.has_bitmask() && bot_mask == m * n + 1 {
            ctx.vload_mask(
                bot_mask,
                Mem {
                    addr: BOT_MASK_PTR,
                    ..Default::default()
                },
            );
        }

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
                        if ctx.is_cplx() {
                            ctx.vadd(dst, dst, i + m * j);
                        } else {
                            ctx.vfmadd(dst, i + m * j, alpha);
                        }
                    } else if !ctx.is_cplx() {
                        ctx.vmul(dst, i + m * j, alpha);
                    } else {
                        ctx.vmov(dst, i + m * j);
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

    pub fn hkernel(ctx: &mut dyn KernelCtx, m: usize, n: usize, triu: bool) {
        fn setup_cs(ctx: &mut dyn KernelCtx, tmp: &[Reg], cs: Reg) {
            for i in 1..tmp.len() {
                ctx.lea(
                    tmp[i],
                    Mem {
                        addr: tmp[i - 1],
                        index: Some(cs),
                        scale: 2,
                        offset: 0,
                    },
                );
            }
        }

        assert!(m <= 4);
        assert!(m <= 4);
        assert!(m * n + m + 4 < ctx.n_regs());

        let tmp0 = Reg::R13;
        let tmp1 = Reg::R14;

        ctx.push(tmp0);
        ctx.push(tmp1);
        ctx.push(DEPTH);
        ctx.push(LHS_PTR);
        ctx.push(RHS_PTR);
        ctx.push(LHS_CS);

        let tmp_lhs = &[LHS_PTR, tmp0];
        let tmp_rhs = &[RHS_PTR, tmp1];
        let tmp_lhs = &tmp_lhs[..1 + (m - 1) / 2];
        let tmp_rhs = &tmp_rhs[..1 + (n - 1) / 2];

        setup_cs(ctx, tmp_lhs, LHS_CS);
        setup_cs(ctx, tmp_rhs, RHS_CS);

        ctx.vzero(0);
        for i in 1..m * n {
            ctx.vmov(i, 0);
        }

        let top_mask = if ctx.has_bitmask() {
            1
        } else {
            m * n + 2 * m + 1
        };
        let bot_mask = if ctx.has_bitmask() {
            2
        } else {
            m * n + 2 * m + 2
        };

        ctx.vload_mask(
            top_mask,
            Mem {
                addr: TOP_MASK_PTR,
                ..Default::default()
            },
        );
        ctx.vload_mask(
            bot_mask,
            Mem {
                addr: BOT_MASK_PTR,
                ..Default::default()
            },
        );

        let size = ctx.unit_size() * if ctx.is_cplx() { 2 } else { 1 };

        let body = |ctx: &mut dyn KernelCtx, conj: bool| {
            let rhs = m * n + 2 * m;
            ctx.writeln(&format!("cmp {DEPTH}, {}", ctx.reg_size() / size,));
            ctx.writeln("jle 333f");
            {
                ctx.writeln(&format!("add {DEPTH}, {}", ctx.reg_size() / size - 1,));
                ctx.writeln(&format!("shr {DEPTH}, {}", (ctx.reg_size() / size).ilog2(),));
                ctx.writeln(&format!("sub {DEPTH}, 2"));

                for i in 0..m {
                    ctx.vmaskload(
                        top_mask,
                        m * n + 2 * i,
                        Mem {
                            addr: tmp_lhs[i / 2],
                            index: [None, Some(LHS_CS)][i % 2],
                            scale: 1,
                            offset: 0,
                        },
                    );
                }
                for i in 0..m {
                    ctx.vre_im(m * n + 2 * i, m * n + 2 * i + 1);
                }
                for j in 0..n {
                    let m = if triu { j + 1 } else { m };

                    ctx.vmaskload(
                        top_mask,
                        rhs,
                        Mem {
                            addr: tmp_rhs[j / 2],
                            index: [None, Some(RHS_CS)][j % 2],
                            scale: 1,
                            offset: 0,
                        },
                    );
                    for i in 0..m {
                        if conj {
                            ctx.vfmadd_conj(i + j * m, m * n + 2 * i, rhs);
                        } else {
                            ctx.vfmadd(i + j * m, m * n + 2 * i, rhs);
                        }
                    }
                    if ctx.is_cplx() {
                        ctx.vswap(rhs);
                        for i in 0..m {
                            if conj {
                                ctx.vfmadd_conj(i + j * m, m * n + 2 * i + 1, rhs);
                            } else {
                                ctx.vfmadd(i + j * m, m * n + 2 * i + 1, rhs);
                            }
                        }
                    }
                }

                for x in tmp_lhs.iter().chain(tmp_rhs) {
                    ctx.writeln(&format!("add {x}, {}", ctx.reg_size()));
                }

                ctx.loop_(0, DEPTH, &mut |ctx| {
                    for i in 0..m {
                        ctx.vload(
                            m * n + 2 * i,
                            Mem {
                                addr: tmp_lhs[i / 2],
                                index: [None, Some(LHS_CS)][i % 2],
                                scale: 1,
                                offset: 0,
                            },
                        );
                    }
                    for i in 0..m {
                        ctx.vre_im(m * n + 2 * i, m * n + 2 * i + 1);
                    }
                    for j in 0..n {
                        let m = if triu { j + 1 } else { m };

                        ctx.vload(
                            rhs,
                            Mem {
                                addr: tmp_rhs[j / 2],
                                index: [None, Some(RHS_CS)][j % 2],
                                scale: 1,
                                offset: 0,
                            },
                        );
                        for i in 0..m {
                            if conj {
                                ctx.vfmadd_conj(i + j * m, m * n + 2 * i, rhs);
                            } else {
                                ctx.vfmadd(i + j * m, m * n + 2 * i, rhs);
                            }
                        }
                        if ctx.is_cplx() {
                            ctx.vswap(rhs);
                            for i in 0..m {
                                if conj {
                                    ctx.vfmadd_conj(i + j * m, m * n + 2 * i + 1, rhs);
                                } else {
                                    ctx.vfmadd(i + j * m, m * n + 2 * i + 1, rhs);
                                }
                            }
                        }
                    }
                    for x in tmp_lhs.iter().chain(tmp_rhs) {
                        ctx.writeln(&format!("add {x}, {}", ctx.reg_size()));
                    }
                });

                for i in 0..m {
                    ctx.vmaskload(
                        bot_mask,
                        m * n + 2 * i,
                        Mem {
                            addr: tmp_lhs[i / 2],
                            index: [None, Some(LHS_CS)][i % 2],
                            scale: 1,
                            offset: 0,
                        },
                    );
                }
                for i in 0..m {
                    ctx.vre_im(m * n + 2 * i, m * n + 2 * i + 1);
                }
                for j in 0..n {
                    let m = if triu { j + 1 } else { m };
                    ctx.vmaskload(
                        bot_mask,
                        rhs,
                        Mem {
                            addr: tmp_rhs[j / 2],
                            index: [None, Some(RHS_CS)][j % 2],
                            scale: 1,
                            offset: 0,
                        },
                    );
                    for i in 0..m {
                        if conj {
                            ctx.vfmadd_conj(i + j * m, m * n + 2 * i, rhs);
                        } else {
                            ctx.vfmadd(i + j * m, m * n + 2 * i, rhs);
                        }
                    }
                    if ctx.is_cplx() {
                        ctx.vswap(rhs);
                        for i in 0..m {
                            if conj {
                                ctx.vfmadd_conj(i + j * m, m * n + 2 * i + 1, rhs);
                            } else {
                                ctx.vfmadd(i + j * m, m * n + 2 * i + 1, rhs);
                            }
                        }
                    }
                }
            }
            ctx.writeln("jmp 334f");
            {
                ctx.writeln("333:");
                ctx.vand_mask(top_mask, bot_mask);
                for i in 0..m {
                    ctx.vmaskload(
                        top_mask,
                        m * n + 2 * i,
                        Mem {
                            addr: tmp_lhs[i / 2],
                            index: [None, Some(LHS_CS)][i % 2],
                            scale: 1,
                            offset: 0,
                        },
                    );
                }
                for i in 0..m {
                    ctx.vre_im(m * n + 2 * i, m * n + 2 * i + 1);
                }
                for j in 0..n {
                    let m = if triu { j + 1 } else { m };
                    ctx.vmaskload(
                        top_mask,
                        rhs,
                        Mem {
                            addr: tmp_rhs[j / 2],
                            index: [None, Some(RHS_CS)][j % 2],
                            scale: 1,
                            offset: 0,
                        },
                    );
                    for i in 0..m {
                        if conj {
                            ctx.vfmadd_conj(i + j * m, m * n + 2 * i, rhs);
                        } else {
                            ctx.vfmadd(i + j * m, m * n + 2 * i, rhs);
                        }
                    }
                    if ctx.is_cplx() {
                        ctx.vswap(rhs);
                        for i in 0..m {
                            if conj {
                                ctx.vfmadd_conj(i + j * m, m * n + 2 * i + 1, rhs);
                            } else {
                                ctx.vfmadd(i + j * m, m * n + 2 * i + 1, rhs);
                            }
                        }
                    }
                }
            }
            ctx.writeln("334:");
        };

        let nbits = ctx.unit_size() * 8;
        let vreg = ctx.vreg();

        if ctx.is_cplx() {
            let sign = m * n + m;
            ctx.branch_bit(
                1,
                3,
                FLAGS,
                &mut |ctx| {
                    body(ctx, true);
                    ctx.writeln(&format!(
                        "vmovupd {vreg}{sign}, [rip + LIBFAER_GEMM_COMPLEX{nbits}_MASK_REAL]"
                    ));
                },
                &mut |ctx| {
                    body(ctx, false);
                    ctx.vzero(sign);
                },
            );

            ctx.branch_bit(
                    1,
                    4,
                    FLAGS,
                    &mut |ctx| {
                        ctx.writeln(&format!(
                            "vorpd {vreg}{sign}, {vreg}{sign}, [rip + LIBFAER_GEMM_COMPLEX{nbits}_MASK_IMAG]"
                        ));
                    },
                    &mut |_| {},
                );

            for i in 0..m * n {
                ctx.vxor(i, sign);
            }
        } else {
            body(ctx, false);
        }

        let tmp = [DST_PTR, LHS_PTR, RHS_PTR, tmp0];
        setup_cs(ctx, &tmp[..tmp_lhs.len()], HORIZONTAL_DST_RS);
        for i in 0..tmp_lhs.len() {
            let tmp = [tmp[i], tmp[i + 2]];
            setup_cs(ctx, &tmp[..tmp_rhs.len()], DST_CS);
        }
        ctx.lea(
            LHS_CS,
            Mem {
                addr: DST_CS,
                index: Some(HORIZONTAL_DST_RS),
                scale: 1,
                offset: 0,
            },
        );

        let alpha = m * n;
        let alpha_imag = m * n + 1;
        let dst = m * n + 2;
        ctx.vbroadcast(
            alpha,
            Mem {
                addr: ALPHA_PTR,
                index: None,
                scale: 0,
                offset: 0,
            },
        );

        if ctx.is_cplx() {
            ctx.vbroadcast(
                alpha_imag,
                Mem {
                    addr: ALPHA_PTR,
                    index: None,
                    scale: 0,
                    offset: ctx.unit_size(),
                },
            );

            for i in 0..m * n {
                ctx.vzero(dst);
                ctx.vfmadd_conj(dst, i, alpha);
                ctx.vswap(i);
                ctx.vfmadd_conj(dst, i, alpha_imag);
                ctx.vmov(i, dst);
            }

            let sign = dst;
            ctx.writeln(&format!(
                "vmovupd {vreg}{sign}, [rip + LIBFAER_GEMM_COMPLEX{nbits}_MASK_REAL]"
            ));
            for i in 0..m * n {
                ctx.vxor(i, sign);
            }
        } else {
            for i in 0..m * n {
                ctx.vmul(i, i, alpha);
            }
        }

        let update = |ctx: &mut dyn KernelCtx, load: bool| {
            for j in 0..n {
                let m = if triu { j + 1 } else { m };
                for i in 0..m {
                    let addr = tmp[i / 2 + 2 * (j / 2)];
                    let (index, scale) = match (i % 2, j % 2) {
                        (0, 0) => (None, 0),
                        (1, 0) => (Some(HORIZONTAL_DST_RS), 1),
                        (0, 1) => (Some(DST_CS), 1),
                        (1, 1) => (Some(LHS_CS), 1),
                        _ => unreachable!(),
                    };
                    let mem = Mem {
                        addr,
                        index,
                        scale,
                        offset: 0,
                    };

                    let instr = match (ctx.is_cplx(), ctx.unit_size()) {
                        (false, 4) => "vmovss",
                        (false, 8) => "vmovsd",
                        (true, 4) => "vmovsd",
                        (true, 8) => "vmovupd",
                        _ => unreachable!(),
                    };

                    if load {
                        ctx.writeln(&format!("{instr} xmm{dst}, {mem}"));
                        ctx.vload(dst, mem);
                        ctx.vadd(dst, dst, i + m * j);
                    } else {
                        ctx.vmov(dst, i + m * j);
                    }

                    ctx.vreduce(dst, i + m * j);
                    ctx.writeln(&format!("{instr} {mem}, xmm{dst}"));
                }
            }
        };

        ctx.branch_bit(
            1, //
            0,
            FLAGS,
            &mut |ctx| update(ctx, true),
            &mut |ctx| update(ctx, false),
        );

        ctx.pop(LHS_CS);
        ctx.pop(RHS_PTR);
        ctx.pop(LHS_PTR);
        ctx.pop(DEPTH);
        ctx.pop(tmp1);
        ctx.pop(tmp0);
        ctx.vzeroupper();
        ctx.ret();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut codegen = String::new();
    let mut bindings = String::new();

    for domain in [Domain::Real, Domain::Cplx] {
        for dtype in [DType::F32, DType::F64] {
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

                        let name = format!("libfaer_gemm_{domain:?}_{dtype:?}_avx_{m}x{n}");
                        writeln!(codegen, ".align 32")?;
                        writeln!(codegen, ".globl {name}")?;
                        writeln!(codegen, "{name}:")?;

                        let mut ctx = x86_64::RealAvx(String::new(), dtype, domain);
                        x86_64::microkernel(&mut ctx, m, n);
                        writeln!(codegen, "{}", ctx.0)?;
                        writeln!(bindings, "pub fn {name}();")?;
                        names.push(name);
                    }
                }

                writeln!(bindings, "}}")?;
                writeln!(
                    bindings,
                    "#[allow(non_upper_case_globals)] pub const UKR_AVX_{domain:?}_{dtype:?}: &[[unsafe extern fn(); {max_m}]; {max_n}] = &"
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
            {
                let max_n = 2;
                let max_m = 2;

                writeln!(bindings, "extern {{")?;
                let mut names = vec![];
                for triu in [false] {
                    for j in 0..max_n {
                        let n = j + 1;
                        for i in 0..max_m {
                            let m = i + 1;

                            let name = format!(
                                "libfaer_gemm_reduce_{domain:?}_{dtype:?}_avx_{m}x{n}{}",
                                if triu { "_tri_up" } else { "" }
                            );
                            writeln!(codegen, ".align 32")?;
                            writeln!(codegen, ".globl {name}")?;
                            writeln!(codegen, "{name}:")?;

                            let mut ctx = x86_64::RealAvx(String::new(), dtype, domain);
                            x86_64::hkernel(&mut ctx, m, n, triu);
                            writeln!(codegen, "{}", ctx.0)?;
                            writeln!(bindings, "pub fn {name}();")?;
                            names.push(name);
                        }
                    }
                }
                writeln!(bindings, "}}")?;
                writeln!(
                    bindings,
                    "#[allow(non_upper_case_globals)] pub const HKR_AVX_{domain:?}_{dtype:?}: &[[unsafe extern fn(); {max_m}]; {max_n}] = &"
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

                        let name = format!("libfaer_gemm_{domain:?}_{dtype:?}_avx512_{m}x{n}");
                        writeln!(codegen, ".align 32")?;
                        writeln!(codegen, ".globl {name}")?;
                        writeln!(codegen, "{name}:")?;

                        let mut ctx = x86_64::RealAvx512(String::new(), dtype, domain);
                        x86_64::microkernel(&mut ctx, m, n);
                        writeln!(codegen, "{}", ctx.0)?;
                        writeln!(bindings, "pub fn {name}();")?;
                        names.push(name);
                    }
                }

                writeln!(bindings, "}}")?;
                writeln!(
                    bindings,
                    "#[allow(non_upper_case_globals)] pub const UKR_AVX512_{domain:?}_{dtype:?}: &[[unsafe extern fn(); {max_m}]; {max_n}] = &"
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

            {
                let max_n = 4;
                let max_m = 4;

                writeln!(bindings, "extern {{")?;
                let mut names = vec![];
                for triu in [false] {
                    for j in 0..max_n {
                        let n = j + 1;
                        for i in 0..max_m {
                            let m = i + 1;

                            let name = format!(
                                "libfaer_gemm_reduce_{domain:?}_{dtype:?}_avx512_{m}x{n}{}",
                                if triu { "_tri_up" } else { "" }
                            );
                            writeln!(codegen, ".align 32")?;
                            writeln!(codegen, ".globl {name}")?;
                            writeln!(codegen, "{name}:")?;

                            let mut ctx = x86_64::RealAvx512(String::new(), dtype, domain);
                            x86_64::hkernel(&mut ctx, m, n, triu);
                            writeln!(codegen, "{}", ctx.0)?;
                            writeln!(bindings, "pub fn {name}();")?;
                            names.push(name);
                        }
                    }
                }
                writeln!(bindings, "}}")?;
                writeln!(
                    bindings,
                    "#[allow(non_upper_case_globals)] pub const HKR_AVX512_{domain:?}_{dtype:?}: &[[unsafe extern fn(); {max_m}]; {max_n}] = &"
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
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("codegen.s");
    fs::write(&dest_path, &codegen)?;

    let dest_path = Path::new(&out_dir).join("bindings.rs");
    fs::write(&dest_path, &bindings)?;

    println!("cargo::rerun-if-changed=build.rs");
    Ok(())
}
