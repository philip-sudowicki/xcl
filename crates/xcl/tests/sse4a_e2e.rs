//! SSE4a's EXTRQ / INSERTQ run as real instructions in a title image, through the exception
//! handler XCL installs. On a CPU without SSE4a (any Intel) each one raises
//! STATUS_ILLEGAL_INSTRUCTION and `eh::emulate_console_only` does it on the XMM registers of the
//! exception's context; on an AMD CPU the hardware runs them. The low quadwords must be what
//! `insn::extrq` / `insn::insertq` give either way, so this checks the emulation end to end on
//! Intel and the reference functions against the hardware on AMD.

use std::sync::{atomic::AtomicBool, Arc, Mutex, Once};
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE};
use xcl::insn::{extrq, insertq};
use xcl::modules::{self, MappedModule};

/// Machine code at the start of a page registered as a title image (the handler emulates only
/// in title code); unregistered and freed on drop.
struct Title(usize);

impl Title {
    fn new(code: &[u8]) -> Title {
        static HANDLER: Once = Once::new();
        HANDLER.call_once(xcl::eh::install_crash_handler);
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            assert!(!page.is_null());
            std::ptr::write_bytes(page, 0xCC, 4096);
            std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
            modules::register_mapped(Arc::new(MappedModule {
                name: "sse4a_e2e.exe".into(),
                path: "sse4a_e2e.exe".into(),
                base: page as usize,
                size: 4096,
                entry_rva: 0,
                is_dll: false,
                exports: Default::default(),
                tls_callbacks: Mutex::new(Vec::new()),
                no_thread_calls: AtomicBool::new(false),
                initialized: AtomicBool::new(true),
                stack_reserve: 0,
            }));
            Title(page as usize)
        }
    }

    fn f1(&self) -> extern "system" fn(u64) -> u64 {
        unsafe { std::mem::transmute(self.0) }
    }
    fn f2(&self) -> extern "system" fn(u64, u64) -> u64 {
        unsafe { std::mem::transmute(self.0) }
    }
    fn f3(&self) -> extern "system" fn(u64, u64, u64) -> u64 {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl Drop for Title {
    fn drop(&mut self) {
        modules::unregister_mapped(self.0);
        unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
    }
}

// movq xmm0, rcx / movq xmm1, rdx / movq xmm2, r8 / movq rax, xmm0 / punpcklqdq xmm1, xmm2 / ret
const XMM0_RCX: [u8; 5] = [0x66, 0x48, 0x0F, 0x6E, 0xC1];
const XMM1_RDX: [u8; 5] = [0x66, 0x48, 0x0F, 0x6E, 0xCA];
const XMM2_R8: [u8; 5] = [0x66, 0x49, 0x0F, 0x6E, 0xD0];
const RAX_XMM0: [u8; 5] = [0x66, 0x48, 0x0F, 0x7E, 0xC0];
const XMM1_HI_XMM2: [u8; 4] = [0x66, 0x0F, 0x6C, 0xCA];
const RET: u8 = 0xC3;

/// Values whose fields differ in every byte.
const VALUES: [u64; 4] = [0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, u64::MAX, 0x8000_0000_0000_0001];

/// (length, index) pairs inside the low quadword, where the result is defined on every AMD CPU:
/// a retail game's 16 at 16, and the edges (0 is a length of 64).
const FIELDS: [(u8, u8); 6] = [(16, 16), (8, 0), (1, 63), (32, 32), (0, 0), (5, 3)];

#[test]
fn insertq_and_extrq_with_immediates() {
    for (len, idx) in FIELDS {
        // insertq xmm0, xmm1, len, idx (F2 0F 78 /r ib ib)
        let code = [&XMM0_RCX[..], &XMM1_RDX, &[0xF2, 0x0F, 0x78, 0xC1, len, idx], &RAX_XMM0, &[RET]].concat();
        let t = Title::new(&code);
        // extrq xmm0, len, idx (66 0F 78 /0 ib ib)
        let code = [&XMM0_RCX[..], &[0x66, 0x0F, 0x78, 0xC0, len, idx], &RAX_XMM0, &[RET]].concat();
        let e = Title::new(&code);
        for a in VALUES {
            for b in VALUES {
                assert_eq!(t.f2()(a, b), insertq(a, b, len, idx), "insertq {a:#x}, {b:#x}, {len}, {idx}");
            }
            assert_eq!(e.f1()(a), extrq(a, len, idx), "extrq {a:#x}, {len}, {idx}");
        }
    }
    // a retail game's insertq xmm0, xmm1, 16, 16, worked out by hand
    let code = [&XMM0_RCX[..], &XMM1_RDX, &[0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], &RAX_XMM0, &[RET]].concat();
    assert_eq!(Title::new(&code).f2()(u64::MAX, 0x1234), 0xFFFF_FFFF_1234_FFFF);
}

#[test]
fn insertq_and_extrq_with_the_field_in_a_register() {
    for (len, idx) in FIELDS {
        let ctl = len as u64 | (idx as u64) << 8;
        // insertq xmm0, xmm1 (F2 0F 79 /r): the field in xmm1's bits 69:64 and 77:72
        let code = [&XMM0_RCX[..], &XMM1_RDX, &XMM2_R8, &XMM1_HI_XMM2, &[0xF2, 0x0F, 0x79, 0xC1], &RAX_XMM0, &[RET]].concat();
        let t = Title::new(&code);
        // extrq xmm0, xmm1 (66 0F 79 /r): the field in xmm1's bits 5:0 and 13:8
        let code = [&XMM0_RCX[..], &XMM1_RDX, &[0x66, 0x0F, 0x79, 0xC1], &RAX_XMM0, &[RET]].concat();
        let e = Title::new(&code);
        for a in VALUES {
            for b in VALUES {
                assert_eq!(t.f3()(a, b, ctl), insertq(a, b, len, idx), "insertq {a:#x}, {b:#x} with {len}, {idx}");
            }
            assert_eq!(e.f2()(a, ctl), extrq(a, len, idx), "extrq {a:#x} with {len}, {idx}");
        }
    }
}

#[test]
fn insertq_on_registers_above_xmm7() {
    // insertq xmm9, xmm12, 8, 40 (REX.R and REX.B); xmm9 and xmm12 are callee-saved, so the
    // function keeps them on the stack
    #[rustfmt::skip]
    let code = [
        0x48, 0x83, 0xEC, 0x20,                         // sub rsp, 32
        0xF3, 0x44, 0x0F, 0x7F, 0x0C, 0x24,             // movdqu [rsp], xmm9
        0xF3, 0x44, 0x0F, 0x7F, 0x64, 0x24, 0x10,       // movdqu [rsp+16], xmm12
        0x66, 0x4C, 0x0F, 0x6E, 0xC9,                   // movq xmm9, rcx
        0x66, 0x4C, 0x0F, 0x6E, 0xE2,                   // movq xmm12, rdx
        0xF2, 0x45, 0x0F, 0x78, 0xCC, 0x08, 0x28,       // insertq xmm9, xmm12, 8, 40
        0x66, 0x4C, 0x0F, 0x7E, 0xC8,                   // movq rax, xmm9
        0xF3, 0x44, 0x0F, 0x6F, 0x0C, 0x24,             // movdqu xmm9, [rsp]
        0xF3, 0x44, 0x0F, 0x6F, 0x64, 0x24, 0x10,       // movdqu xmm12, [rsp+16]
        0x48, 0x83, 0xC4, 0x20,                         // add rsp, 32
        RET,
    ];
    let t = Title::new(&code);
    for a in VALUES {
        for b in VALUES {
            assert_eq!(t.f2()(a, b), insertq(a, b, 8, 40), "insertq {a:#x}, {b:#x}, 8, 40 on xmm9, xmm12");
        }
    }
}

#[test]
fn emulated_insertq_and_extrq_zero_the_high_quadword() {
    // undefined in AMD's manual and what a Zen 5 does; older AMD CPUs may differ, so this is
    // only checked where the handler does the work
    if std::arch::is_x86_feature_detected!("sse4a") {
        println!("SKIPPED emulated_insertq_and_extrq_zero_the_high_quadword: this CPU has SSE4a");
        return;
    }
    // xmm0 = [rcx, r8]; xmm1 = rdx; <op>; rax = xmm0's high quadword (punpckhqdq xmm0, xmm0)
    let high = |op: &[u8]| {
        [&XMM0_RCX[..], &[0x66, 0x49, 0x0F, 0x6E, 0xD8], &[0x66, 0x0F, 0x6C, 0xC3], &XMM1_RDX, op, &[0x66, 0x0F, 0x6D, 0xC0], &RAX_XMM0, &[RET]].concat()
    };
    // (movq xmm3, r8; punpcklqdq xmm0, xmm3 above)
    for op in [&[0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10][..], &[0x66, 0x0F, 0x78, 0xC0, 0x08, 0x04]] {
        let t = Title::new(&high(op));
        assert_eq!(t.f3()(1, 2, u64::MAX), 0, "code bytes {op:02x?}");
    }
}
