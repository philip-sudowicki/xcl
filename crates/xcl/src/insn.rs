//! Instructions the console's CPU (Zen 2) has and a PC's may lack – a title raises
//! STATUS_ILLEGAL_INSTRUCTION on them on an Intel CPU before Ice Lake / Alder Lake (CLWB,
//! SHA) or on any Intel CPU (CLZERO, RDPRU, MONITORX / MWAITX and SSE4a's EXTRQ / INSERTQ,
//! which are AMD's). The exception handler (`eh`) emulates the ones decoded here and steps over
//! them.

/// A console-only instruction at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// `clzero` loop: `clzero; add rax, 0x40; cmp rax, <reg>; jb -12` – zero `[rax..reg)`
    ClzeroLoop { end_reg: u8 },
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips
    Monitorx,
    Mwaitx,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a `extrq xmm, imm8 (length), imm8 (index)` / `extrq xmm1, xmm2` (`src` Some: the
    /// length and index in xmm2's bits 5:0 and 13:8): a bit field of the low quadword, moved
    /// down and zero-extended (the high quadword zeroed). Registers are XMM numbers 0..15.
    Extrq { dst: u8, src: Option<u8>, len: u8, idx: u8 },
    /// SSE4a `insertq xmm1, xmm2, imm8 (length), imm8 (index)` / `insertq xmm1, xmm2` (`reg`:
    /// the length and index in xmm2's bits 69:64 and 77:72): the low bits of xmm2 put into a
    /// bit field of xmm1's low quadword.
    Insertq { dst: u8, src: u8, reg: bool, len: u8, idx: u8 },
    /// A SHA extension instruction (Intel has them from Ice Lake / Rocket Lake on): `dst` is
    /// xmm1, also the first source, `src` xmm2 / m128 ([`sha`] runs it).
    Sha { op: ShaOp, dst: u8, src: Operand },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

/// The SHA instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaOp {
    /// `sha1rnds4 xmm1, xmm2/m128, imm8`: four SHA-1 rounds of the group imm8 (0..3)
    Sha1Rnds4(u8),
    Sha1Nexte,
    Sha1Msg1,
    Sha1Msg2,
    /// `sha256rnds2 xmm1, xmm2/m128, <XMM0>`: two SHA-256 rounds, their W + K in XMM0's low
    /// quadword
    Sha256Rnds2,
    Sha256Msg1,
    Sha256Msg2,
}

/// An XMM register or a memory operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Xmm(u8),
    Mem(MemRef),
}

/// A ModRM memory operand with 64-bit addressing: `base + index * scale + disp`, or `rip + disp`
/// (`rip`: the address of the next instruction). Registers in CONTEXT order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemRef {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8,
    pub disp: i32,
    pub rip: bool,
}

impl MemRef {
    /// The address, with `regs` the 16 general registers (CONTEXT order) and `next` the address
    /// of the instruction after this one.
    pub fn address(&self, regs: &[u64; 16], next: u64) -> u64 {
        let mut a = if self.rip { next } else { self.base.map_or(0, |b| regs[b as usize & 15]) };
        if let Some(x) = self.index {
            a = a.wrapping_add(regs[x as usize & 15].wrapping_mul(self.scale as u64));
        }
        a.wrapping_add(self.disp as i64 as u64)
    }
}

/// A SHA instruction on XMM lanes (lane 0 is bits 31:0): `a` is xmm1, `b` xmm2 / m128 and `xmm0`
/// XMM0 (SHA256RNDS2's W + K); the result is xmm1's new value. As Intel's SDM gives them.
pub fn sha(op: ShaOp, a: [u32; 4], b: [u32; 4], xmm0: [u32; 4]) -> [u32; 4] {
    let s0 = |w: u32| w.rotate_right(7) ^ w.rotate_right(18) ^ (w >> 3);
    let s1 = |w: u32| w.rotate_right(17) ^ w.rotate_right(19) ^ (w >> 10);
    match op {
        ShaOp::Sha1Rnds4(f) => {
            let k = [0x5A82_7999u32, 0x6ED9_EBA1, 0x8F1B_BCDC, 0xCA62_C1D6][f as usize & 3];
            let fun = |b: u32, c: u32, d: u32| match f & 3 {
                0 => (b & c) ^ (!b & d),
                2 => (b & c) ^ (b & d) ^ (c & d),
                _ => b ^ c ^ d,
            };
            // A B C D from lanes 3..0, W0 + E then W1..W3 from b's lanes 3..0 (E is 0 in round 0)
            let (mut s, mut e) = ([a[3], a[2], a[1], a[0]], 0u32);
            for w in [b[3], b[2], b[1], b[0]] {
                let [sa, sb, sc, sd] = s;
                let t = fun(sb, sc, sd).wrapping_add(sa.rotate_left(5)).wrapping_add(w).wrapping_add(e).wrapping_add(k);
                e = sd;
                s = [t, sa, sb.rotate_left(30), sc];
            }
            [s[3], s[2], s[1], s[0]]
        }
        ShaOp::Sha1Nexte => [b[0], b[1], b[2], b[3].wrapping_add(a[3].rotate_left(30))],
        ShaOp::Sha1Msg1 => [b[2] ^ a[0], b[3] ^ a[1], a[0] ^ a[2], a[1] ^ a[3]],
        ShaOp::Sha1Msg2 => {
            let w16 = (a[3] ^ b[2]).rotate_left(1);
            let w17 = (a[2] ^ b[1]).rotate_left(1);
            let w18 = (a[1] ^ b[0]).rotate_left(1);
            let w19 = (a[0] ^ w16).rotate_left(1);
            [w19, w18, w17, w16]
        }
        ShaOp::Sha256Rnds2 => {
            // A B E F in b's lanes 3, 2, 1, 0; C D G H in a's
            let [mut h, mut g, mut d, mut c] = a;
            let [mut f, mut e, mut bb, mut aa] = b;
            for wk in [xmm0[0], xmm0[1]] {
                let ch = (e & f) ^ (!e & g);
                let maj = (aa & bb) ^ (aa & c) ^ (bb & c);
                let sum0 = aa.rotate_right(2) ^ aa.rotate_right(13) ^ aa.rotate_right(22);
                let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let t = ch.wrapping_add(sum1).wrapping_add(wk).wrapping_add(h);
                (h, g, f) = (g, f, e);
                e = t.wrapping_add(d);
                (d, c, bb) = (c, bb, aa);
                aa = t.wrapping_add(maj).wrapping_add(sum0);
            }
            [f, e, bb, aa]
        }
        ShaOp::Sha256Msg1 => [
            a[0].wrapping_add(s0(a[1])),
            a[1].wrapping_add(s0(a[2])),
            a[2].wrapping_add(s0(a[3])),
            a[3].wrapping_add(s0(b[0])),
        ],
        ShaOp::Sha256Msg2 => {
            let w16 = a[0].wrapping_add(s1(b[2]));
            let w17 = a[1].wrapping_add(s1(b[3]));
            let w18 = a[2].wrapping_add(s1(w16));
            let w19 = a[3].wrapping_add(s1(w17));
            [w16, w17, w18, w19]
        }
    }
}

/// A field length of SSE4a's six bits: 0 is 64.
fn field_len(len: u8) -> u32 {
    match len & 63 {
        0 => 64,
        n => n as u32,
    }
}

/// EXTRQ on a low quadword: `len` bits from bit `idx`, zero-extended (a field past bit 63 is
/// undefined on the console; here it is what a shift gives).
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let field = v.checked_shr(idx).unwrap_or(0);
    if len >= 64 {
        field
    } else {
        field & ((1u64 << len) - 1)
    }
}

/// INSERTQ on a low quadword: the low `len` bits of `src` put at bit `idx` of `dst`.
pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let ones = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let mask = ones.checked_shl(idx).unwrap_or(0);
    (dst & !mask) | ((src & ones).checked_shl(idx).unwrap_or(0) & mask)
}

/// The length of a ModRM-encoded operand (ModRM, SIB and displacement) at `b[0]`, None when the
/// bytes run out.
fn modrm_len(b: &[u8]) -> Option<usize> {
    let modrm = *b.first()?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return Some(1);
    }
    let mut len = 1;
    let mut base = rm;
    if rm == 4 {
        base = *b.get(1)? & 7;
        len += 1;
    }
    len += match md {
        0 if base == 5 => 4, // disp32 (rip-relative, or a SIB without a base)
        0 => 0,
        1 => 1,
        _ => 4,
    };
    (b.len() >= len).then_some(len)
}

/// The ModRM operand at `b[0]` with 64-bit addressing (`rex`: the REX byte, 0 without one) and
/// its length (ModRM, SIB and displacement): an XMM register for mod 3.
fn operand(b: &[u8], rex: u8) -> Option<(Operand, usize)> {
    let len = modrm_len(b)?;
    let (md, rm) = (b[0] >> 6, b[0] & 7);
    let (rex_b, rex_x) = ((rex & 1) << 3, (rex & 2) << 2);
    if md == 3 {
        return Some((Operand::Xmm(rm | rex_b), 1));
    }
    let sib = (rm == 4).then(|| b[1]);
    let d = &b[1 + sib.is_some() as usize..len];
    let disp = match *d {
        [] => 0,
        [d8] => d8 as i8 as i32,
        [d0, d1, d2, d3] => i32::from_le_bytes([d0, d1, d2, d3]),
        _ => return None,
    };
    let m = match sib {
        None if md == 0 && rm == 5 => MemRef { base: None, index: None, scale: 1, disp, rip: true },
        None => MemRef { base: Some(rm | rex_b), index: None, scale: 1, disp, rip: false },
        // index 4 (rsp) is none; base 5 under mod 0 is none, with a disp32
        Some(s) => {
            let index = ((s >> 3) & 7) | rex_x;
            let base = s & 7;
            MemRef {
                base: (md != 0 || base != 5).then_some(base | rex_b),
                index: (index != 4).then_some(index),
                scale: 1 << (s >> 6),
                disp,
                rip: false,
            }
        }
    };
    Some((Operand::Mem(m), len))
}

/// A SHA instruction with its ModRM at `code[i + 3]` (`i`: where its 0F is), an imm8 after the
/// operand for SHA1RNDS4. A memory operand only with 64-bit addressing and without FS / GS.
fn decode_sha(code: &[u8], i: usize, rex: u8, op: ShaOp) -> Option<(ConsoleOnly, usize)> {
    let m = *code.get(i + 3)?;
    let (src, len) = operand(&code[i + 3..], rex)?;
    if matches!(src, Operand::Mem(_)) && code[..i].iter().any(|b| matches!(b, 0x64 | 0x65 | 0x67)) {
        return None;
    }
    let dst = ((m >> 3) & 7) | (rex & 4) << 1;
    let mut end = i + 3 + len;
    let op = match op {
        ShaOp::Sha1Rnds4(_) => {
            end += 1;
            ShaOp::Sha1Rnds4(*code.get(end - 1)? & 3)
        }
        op => op,
    };
    Some((ConsoleOnly::Sha { op, dst, src }, end))
}

/// The console-only instruction at the start of `code` and its length.
pub fn decode(code: &[u8]) -> Option<(ConsoleOnly, usize)> {
    let mut i = 0;
    // `rep`: the last of F2 / F3, which decides between them (and over 66) for SSE4a
    let (mut p66, mut pf3, mut rep, mut rex) = (false, false, 0u8, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF2 => rep = 0xF2,
            0xF3 => {
                pf3 = true;
                rep = 0xF3;
            }
            0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 | 0x67 => {}
            // REX counts only right before the opcode: one before a legacy prefix is ignored
            0x40..=0x4F => {
                rex = b;
                i += 1;
                continue;
            }
            _ => break,
        }
        rex = 0;
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => {
            // clzero (0F 01 FC); add rax, 0x40 (48 83 C0 40); cmp rax, <reg> ([48|4C] 39 modrm); jb -12 (72 F4)
            // (not after an 0x67: that CLZERO is at EAX, and the jb goes back to the 0F, past the
            // prefix – it is stepped alone, the loop is taken from the next one)
            if code.len() >= i + 12
                && !code[..i].contains(&0x67)
                && code[i + 3..i + 7] == [0x48, 0x83, 0xC0, 0x40]
                && (code[i + 7] == 0x48 || code[i + 7] == 0x4C)
                && code[i + 8] == 0x39
                && (code[i + 9] & 0xC7) == 0xC0
                && code[i + 10..i + 12] == [0x72, 0xF4]
            {
                let rex_r = code[i + 7] == 0x4C;
                let modrm = code[i + 9];
                let reg_idx = ((modrm >> 3) & 7) | if rex_r { 8 } else { 0 };
                if reg_idx != 0 {
                    return Some((ConsoleOnly::ClzeroLoop { end_reg: reg_idx }, i + 12));
                }
            }
            Some((ConsoleOnly::Clzero, i + 3))
        }
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // SSE4a: register operands only (ModRM mod 3); REX.R extends reg, REX.B rm
        // (the last of F2 / F3 decides: F2 INSERTQ, F3 #UD; neither with 66: EXTRQ)
        (0x78 | 0x79, Some(m)) if (rep == 0xF2 || (rep == 0 && p66)) && m >> 6 == 3 => {
            let reg = ((m >> 3) & 7) | (rex & 4) << 1;
            let rm = (m & 7) | (rex & 1) << 3;
            let imm = |k: usize| code.get(i + 3 + k).copied();
            match (code[i + 1], rep == 0xF2) {
                // extrq xmm(rm), imm8, imm8 – the ModRM reg field is /0
                (0x78, false) if (m >> 3) & 7 == 0 => Some((ConsoleOnly::Extrq { dst: rm, src: None, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, false) => Some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, i + 3)),
                (0x78, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, i + 3)),
                _ => None,
            }
        }
        // SHA: NP 0F 38 C8..CD /r and NP 0F 3A CC /r ib (with 66 / F2 / F3 they are other
        // instructions or #UD)
        (0x38, Some(o @ 0xC8..=0xCD)) if !p66 && rep == 0 => {
            let op = [ShaOp::Sha1Nexte, ShaOp::Sha1Msg1, ShaOp::Sha1Msg2, ShaOp::Sha256Rnds2, ShaOp::Sha256Msg1, ShaOp::Sha256Msg2];
            decode_sha(code, i, rex, op[(o - 0xC8) as usize])
        }
        (0x3A, Some(0xCC)) if !p66 && rep == 0 => decode_sha(code, i, rex, ShaOp::Sha1Rnds4(0)),
        // 66 0F AE /6 with a memory operand (without 66 it is xsaveopt)
        (0xAE, Some(m)) if p66 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        _ => None,
    }
}

/// `code` as hex, for a log line.
pub fn hex(code: &[u8]) -> String {
    code.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXTRQ / INSERTQ as AMD's manual gives them: a length of 0 is 64.
    #[test]
    fn sse4a_bit_fields() {
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 8, 16), 0xBC);
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 0, 0), 0x1234_5678_9ABC_DEF0);
        assert_eq!(extrq(0x8000_0000_0000_0000, 1, 63), 1);
        assert_eq!(insertq(0xFFFF_FFFF_FFFF_FFFF, 0xAB, 8, 8), 0xFFFF_FFFF_FFFF_ABFF);
        assert_eq!(insertq(0, 0x1FF, 4, 60), 0xF000_0000_0000_0000);
        assert_eq!(insertq(0x1111, 0xDEAD_BEEF_0000_0000, 0, 0), 0xDEAD_BEEF_0000_0000);
        for (v, len, idx) in [(0x0123_4567_89AB_CDEFu64, 12u8, 20u8), (u64::MAX, 63, 1), (0x55, 3, 0)] {
            assert_eq!(extrq(insertq(0, v, len, idx), len, idx), extrq(v, len, 0));
        }
    }

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
        // clzero loop against r10 (4C 39 D0 = cmp rax, r10)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 10 }, 12))
        );
        // clzero loop against rdx (48 39 D0 = cmp rax, rdx)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 2 }, 12))
        );
        // the same loop after an address-size prefix: that one CLZERO alone
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 4)));
        // cmp with non-rax rm, memory operand, or cmp rax, rax is not a ClzeroLoop
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xCA, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0x10, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xC0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((ConsoleOnly::Rdpru, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((ConsoleOnly::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((ConsoleOnly::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x0B]), Some((ConsoleOnly::Ud2, 2)));
        // clwb [rax]; clwb [rcx+0x40]; clwb [rsp+8]; clwb [rip+disp32]; clwb [r8]
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x71, 0x40]), Some((ConsoleOnly::Clwb, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some((ConsoleOnly::Clwb, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x35, 1, 2, 3, 4]), Some((ConsoleOnly::Clwb, 8)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 5)));
        // rdpid rax; rdpid r9
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, 0xF8]), Some((ConsoleOnly::Rdpid(0), 4)));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((ConsoleOnly::Rdpid(9), 5)));
        // not console-only: clflushopt (/7), xsaveopt (no 66), a plain mov, truncated bytes
        // SSE4a, as a retail game has it (#216: insertq xmm0, xmm1 with length / index bytes)
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 8, idx: 16 }, 6)));
        assert_eq!(decode(&[0xF2, 0x45, 0x0F, 0x79, 0xC1]), Some((ConsoleOnly::Insertq { dst: 8, src: 9, reg: true, len: 0, idx: 0 }, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xC2, 0x04, 0x08]), Some((ConsoleOnly::Extrq { dst: 2, src: None, len: 4, idx: 8 }, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0x79, 0xCA]), Some((ConsoleOnly::Extrq { dst: 1, src: Some(2), len: 0, idx: 0 }, 4)));
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08]), None, "an immediate missing");
        assert_eq!(decode(&[0x0F, 0x78, 0xC1, 0x08, 0x10]), None, "VMREAD without a prefix is not SSE4a");
        // prefixes as a Zen 5 runs them (Kalmarv on #209): the last of F2 / F3 decides, 66 next
        // to them does not; 0x67 and a REX before a legacy prefix are ignored
        let insertq = Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 7));
        for code in [[0x66, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF2, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x67, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x41, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]] {
            assert_eq!(decode(&code), insertq, "{code:02x?}");
        }
        for code in [[0xF2, 0xF3, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x66, 0xF3, 0x0F, 0x78, 0xC0, 0x10, 0x10]] {
            assert_eq!(decode(&code), None, "#UD on the CPU: {code:02x?}");
        }
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }

    /// SHA instructions as GNU as 2.42 encodes them, register and memory forms.
    #[test]
    fn decodes_sha() {
        let mem = |base, index, scale, disp| Operand::Mem(MemRef { base, index, scale, disp, rip: false });
        let sha = |op, dst, src, len| Some((ConsoleOnly::Sha { op, dst, src }, len));
        use ShaOp::*;
        // sha1rnds4 xmm1, xmm2, 3; sha1rnds4 xmm9, [rip+0x100], 1
        assert_eq!(decode(&[0x0F, 0x3A, 0xCC, 0xCA, 0x03]), sha(Sha1Rnds4(3), 1, Operand::Xmm(2), 5));
        let rip = MemRef { base: None, index: None, scale: 1, disp: 0x100, rip: true };
        assert_eq!(decode(&[0x44, 0x0F, 0x3A, 0xCC, 0x0D, 0x00, 0x01, 0x00, 0x00, 0x01]), sha(Sha1Rnds4(1), 9, Operand::Mem(rip), 10));
        // sha1nexte xmm3, xmm12; sha1msg1 xmm0, [rax]; sha1msg2 xmm15, [rsp+8]; sha256rnds2 xmm1, xmm2
        assert_eq!(decode(&[0x41, 0x0F, 0x38, 0xC8, 0xDC]), sha(Sha1Nexte, 3, Operand::Xmm(12), 5));
        assert_eq!(decode(&[0x0F, 0x38, 0xC9, 0x00]), sha(Sha1Msg1, 0, mem(Some(0), None, 1, 0), 4));
        assert_eq!(decode(&[0x44, 0x0F, 0x38, 0xCA, 0x7C, 0x24, 0x08]), sha(Sha1Msg2, 15, mem(Some(4), None, 1, 8), 7));
        assert_eq!(decode(&[0x0F, 0x38, 0xCB, 0xCA]), sha(Sha256Rnds2, 1, Operand::Xmm(2), 4));
        // sha256msg1 xmm4, [r13+rcx*4-0x10]; sha256msg2 xmm5, [rbx+r12*8+0x12345678]
        let m1 = mem(Some(13), Some(1), 4, -0x10);
        let m2 = mem(Some(3), Some(12), 8, 0x1234_5678);
        assert_eq!(decode(&[0x41, 0x0F, 0x38, 0xCC, 0x64, 0x8D, 0xF0]), sha(Sha256Msg1, 4, m1, 7));
        assert_eq!(decode(&[0x42, 0x0F, 0x38, 0xCD, 0xAC, 0xE3, 0x78, 0x56, 0x34, 0x12]), sha(Sha256Msg2, 5, m2, 10));
        // sha256msg1 xmm6, [r12] (a SIB without an index); sha256msg2 xmm7, [r13+0]; sha1msg1 xmm2, [0x1000]
        assert_eq!(decode(&[0x41, 0x0F, 0x38, 0xCC, 0x34, 0x24]), sha(Sha256Msg1, 6, mem(Some(12), None, 1, 0), 6));
        assert_eq!(decode(&[0x41, 0x0F, 0x38, 0xCD, 0x7D, 0x00]), sha(Sha256Msg2, 7, mem(Some(13), None, 1, 0), 6));
        assert_eq!(decode(&[0x0F, 0x38, 0xC9, 0x14, 0x25, 0x00, 0x10, 0x00, 0x00]), sha(Sha1Msg1, 2, mem(None, None, 1, 0x1000), 9));
        // FS / GS and 32-bit addressing are not emulated for a memory operand; a register one has no address
        assert_eq!(decode(&[0x64, 0x0F, 0x38, 0xC9, 0x00]), None);
        assert_eq!(decode(&[0x67, 0x0F, 0x38, 0xC9, 0x00]), None);
        assert_eq!(decode(&[0x64, 0x0F, 0x38, 0xC9, 0xC1]), sha(Sha1Msg1, 0, Operand::Xmm(1), 5));
        // with 66 / F3 they are not SHA; truncated: the imm8, the disp32
        assert_eq!(decode(&[0x66, 0x0F, 0x38, 0xC8, 0xC1]), None);
        assert_eq!(decode(&[0xF3, 0x0F, 0x38, 0xC8, 0xC1]), None);
        assert_eq!(decode(&[0x0F, 0x3A, 0xCC, 0xCA]), None);
        assert_eq!(decode(&[0x0F, 0x38, 0xC9, 0x14, 0x25, 0x00, 0x10]), None);
        // the addresses, with register k holding 0x10000 * (k + 1)
        let regs: [u64; 16] = std::array::from_fn(|k| 0x1_0000 * (k as u64 + 1));
        let addr = |o: Operand| match o {
            Operand::Mem(m) => m.address(&regs, 0x40_0000),
            Operand::Xmm(_) => unreachable!(),
        };
        assert_eq!(addr(m1), 0xE_0000 + 4 * 0x2_0000 - 0x10);
        assert_eq!(addr(m2), 0x4_0000 + 8 * 0xD_0000 + 0x1234_5678);
        assert_eq!(addr(Operand::Mem(rip)), 0x40_0100);
        assert_eq!(addr(mem(None, None, 1, 0x1000)), 0x1000);
    }

    /// xorshift64*: test inputs
    fn rng(s: &mut u64) -> u32 {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    const SHA_OPS: [ShaOp; 10] = [
        ShaOp::Sha1Rnds4(0), ShaOp::Sha1Rnds4(1), ShaOp::Sha1Rnds4(2), ShaOp::Sha1Rnds4(3), ShaOp::Sha1Nexte,
        ShaOp::Sha1Msg1, ShaOp::Sha1Msg2, ShaOp::Sha256Rnds2, ShaOp::Sha256Msg1, ShaOp::Sha256Msg2,
    ];

    /// `n` random inputs from `seed`: xmm1, xmm2, XMM0
    fn sha_inputs(seed: u64, n: usize) -> impl Iterator<Item = [[u32; 4]; 3]> {
        let mut s = 0x5EED_0000_0000_0001 ^ seed;
        (0..n).map(move |_| {
            let mut v = [[0u32; 4]; 3];
            v.iter_mut().flatten().for_each(|x| *x = rng(&mut s));
            v
        })
    }

    /// The emulated SHA instructions against what a CPU with them returned (an Intel Xeon with
    /// SHA-NI, Linux): an FNV-1a hash of 256 results of each, so a PC without SHA checks them too.
    #[test]
    fn sha_matches_recorded_cpu_results() {
        let want: [u64; 10] = [
            0xDF4F_D800_7FFF_C390, 0x0C71_5E2F_4FF5_C574, 0x984C_4739_DE9E_93CB, 0xF78F_0DA0_7479_C944, 0x900E_8D9F_6EC0_23DD,
            0xC498_F328_AEBD_D36F, 0xE447_23C7_D074_00F9, 0x7475_CEBE_4F99_8D18, 0xFF6B_5B00_D9E8_75F3, 0x8C82_15BD_D1A5_4E4B,
        ];
        for (k, op) in SHA_OPS.into_iter().enumerate() {
            let mut h = 0xCBF2_9CE4_8422_2325u64;
            for [a, b, x0] in sha_inputs(k as u64, 256) {
                for w in sha(op, a, b, x0) {
                    h = (h ^ w as u64).wrapping_mul(0x100_0000_01B3);
                }
            }
            assert_eq!(h, want[k], "{op:?}");
        }
    }

    /// The emulation against this CPU's own SHA instructions, when it has them (AMD's; Intel's
    /// from Ice Lake / Rocket Lake on).
    #[test]
    fn sha_matches_this_cpu() {
        if !std::arch::is_x86_feature_detected!("sha") {
            return;
        }
        #[target_feature(enable = "sha")]
        unsafe fn cpu(op: ShaOp, a: [u32; 4], b: [u32; 4], x0: [u32; 4]) -> [u32; 4] {
            use std::arch::x86_64::*;
            let (a, b, k): (__m128i, __m128i, __m128i) = (std::mem::transmute(a), std::mem::transmute(b), std::mem::transmute(x0));
            std::mem::transmute(match op {
                ShaOp::Sha1Rnds4(0) => _mm_sha1rnds4_epu32(a, b, 0),
                ShaOp::Sha1Rnds4(1) => _mm_sha1rnds4_epu32(a, b, 1),
                ShaOp::Sha1Rnds4(2) => _mm_sha1rnds4_epu32(a, b, 2),
                ShaOp::Sha1Rnds4(_) => _mm_sha1rnds4_epu32(a, b, 3),
                ShaOp::Sha1Nexte => _mm_sha1nexte_epu32(a, b),
                ShaOp::Sha1Msg1 => _mm_sha1msg1_epu32(a, b),
                ShaOp::Sha1Msg2 => _mm_sha1msg2_epu32(a, b),
                ShaOp::Sha256Rnds2 => _mm_sha256rnds2_epu32(a, b, k),
                ShaOp::Sha256Msg1 => _mm_sha256msg1_epu32(a, b),
                ShaOp::Sha256Msg2 => _mm_sha256msg2_epu32(a, b),
            })
        }
        for (k, op) in SHA_OPS.into_iter().enumerate() {
            for [a, b, x0] in sha_inputs(0x100 + k as u64, 10_000) {
                assert_eq!(sha(op, a, b, x0), unsafe { cpu(op, a, b, x0) }, "{op:?} on {a:08x?} {b:08x?} {x0:08x?}");
            }
        }
    }
}
