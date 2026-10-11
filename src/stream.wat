;; The CPU stream kernel in WebAssembly SIMD: the eight chunk lanes of one group as two halves
;; of four lanes, each word of a half one v128. tools/embed.ts assembles it into
;; src/stream_wasm.ts. stream.ts falls back to its JavaScript kernel where WebAssembly SIMD is
;; missing, with the same words.
;;
;; Memory: the state at 0, the round constants at 288 and the output rows at 512, 128 bytes
;; each, lane l's block at byte 16 l of its row. The state is the lanes, word k of lane l at
;; 4 (8 k + l), so word k of lanes 0 to 3 is the v128 at 32 k and of lanes 4 to 7 the one at
;; 32 k + 16, then the key at 256, the low and high word of chunk counter c of lane 0 at 272
;; and 276, the row j inside the group at 280 and the chunk length K at 284. The Float64
;; exponential map of exp64 has its block of 2048 doubles at 33280 and its constants at 49664.
(module
  (memory (export "memory") 1)
  (global (export "out") i32 (i32.const 512))
  (global (export "capacity") i32 (i32.const 256))
  (data (i32.const 288)
    "\b7\c1\7c\d1" "\94\0a\22\a7" "\e8\ab\13\fe" "\e0\6e\9a\fa"
    "\cc\4a\b1\ed" "\20\c8\21\9e" "\d5\b1\28\ff" "\b0\e2\5d\ef")

  ;; The step T is pasted for each half and each use, since a call per step costs more than
  ;; the step on engines that do not inline.

  ;; seed(): the lanes of chunks c + l through F, at row 0 of their group.
  (func $seed (export "seed")
    (local $r i32) (local $c v128) (local $rc v128) (local $one v128) (local $weyl v128)
    (local $m0 v128) (local $m1 v128) (local $lo0 v128) (local $lo1 v128)
    (local $hi0 v128) (local $hi1 v128) (local $n0 v128) (local $n1 v128) (local $n2 v128)
    (local $n3 v128)
    (local $ao0 v128) (local $ao1 v128) (local $ao2 v128) (local $ao3 v128)
    (local $ah0 v128) (local $ah1 v128) (local $ah2 v128) (local $ah3 v128)
    (local $bo0 v128) (local $bo1 v128) (local $bo2 v128) (local $bo3 v128)
    (local $bh0 v128) (local $bh1 v128) (local $bh2 v128) (local $bh3 v128)
    (local.set $one (i32x4.splat (i32.const 1)))
    (local.set $weyl (i32x4.splat (i32.const 0x9e3779b9)))
    (local.set $c (v128.load32_splat offset=272 (i32.const 0)))
    (local.set $ao0 (i32x4.add (local.get $c) (v128.const i32x4 0 1 2 3)))
    (local.set $bo0 (i32x4.add (local.get $c) (v128.const i32x4 4 5 6 7)))
    (local.set $ao1 (v128.load32_splat offset=276 (i32.const 0)))
    (local.set $ao2 (i32x4.splat (i32.const 0x9e3779b9)))
    (local.set $ao3 (i32x4.splat (i32.const 0x94d049bb)))
    (local.set $ah0 (v128.load32_splat offset=256 (i32.const 0)))
    (local.set $ah1 (v128.load32_splat offset=260 (i32.const 0)))
    (local.set $ah2 (v128.load32_splat offset=264 (i32.const 0)))
    (local.set $ah3 (v128.load32_splat offset=268 (i32.const 0)))
    (local.set $bo1 (local.get $ao1))
    (local.set $bo2 (local.get $ao2))
    (local.set $bo3 (local.get $ao3))
    (local.set $bh0 (local.get $ah0))
    (local.set $bh1 (local.get $ah1))
    (local.set $bh2 (local.get $ah2))
    (local.set $bh3 (local.get $ah3))
    (loop $round
      (local.set $rc (v128.load32_splat offset=288 (i32.shl (local.get $r) (i32.const 2))))
      ;; T on half a
      (local.set $m0 (v128.or (local.get $ah0) (local.get $one)))
      (local.set $m1 (v128.or (local.get $ah1) (local.get $one)))
      (local.set $lo0 (i32x4.mul (local.get $ao0) (local.get $m0)))
      (local.set $lo1 (i32x4.mul (local.get $ao2) (local.get $m1)))
      (local.set $hi0 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $ao0) (local.get $m0))
        (i64x2.extmul_high_i32x4_u (local.get $ao0) (local.get $m0))))
      (local.set $hi1 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $ao2) (local.get $m1))
        (i64x2.extmul_high_i32x4_u (local.get $ao2) (local.get $m1))))
      (local.set $n0 (v128.xor (local.get $ao1) (v128.xor (local.get $hi1) (local.get $lo1))))
      (local.set $n1 (v128.xor (local.get $ah2) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo1) (local.get $lo1))))
      (local.set $n2 (v128.xor (local.get $ao3) (v128.xor (local.get $hi0) (local.get $lo0))))
      (local.set $n3 (v128.xor (local.get $ah3) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo0) (local.get $lo0))))
      (local.set $ah0 (v128.xor (local.get $ah0)
        (v128.or (i32x4.shl (local.get $ah1) (i32.const 7)) (i32x4.shr_u (local.get $ah1) (i32.const 25)))))
      (local.set $ah1 (v128.xor (local.get $ah1)
        (v128.or (i32x4.shl (local.get $ah2) (i32.const 13)) (i32x4.shr_u (local.get $ah2) (i32.const 19)))))
      (local.set $ah2 (v128.xor (local.get $ah2)
        (v128.or (i32x4.shl (local.get $ah3) (i32.const 22)) (i32x4.shr_u (local.get $ah3) (i32.const 10)))))
      (local.set $ah3 (v128.xor (local.get $ah3)
        (v128.or (i32x4.shl (local.get $ah0) (i32.const 3)) (i32x4.shr_u (local.get $ah0) (i32.const 29)))))
      ;; The round constant goes on the new o0, then the halves swap.
      (local.set $ao0 (v128.xor (i32x4.add (local.get $ah0) (local.get $weyl)) (local.get $n0)))
      (local.set $ah0 (v128.xor (local.get $n0) (local.get $rc)))
      (local.set $ao1 (local.get $ah1))
      (local.set $ao2 (local.get $ah2))
      (local.set $ao3 (local.get $ah3))
      (local.set $ah1 (local.get $n1))
      (local.set $ah2 (local.get $n2))
      (local.set $ah3 (local.get $n3))
      ;; T on half b
      (local.set $m0 (v128.or (local.get $bh0) (local.get $one)))
      (local.set $m1 (v128.or (local.get $bh1) (local.get $one)))
      (local.set $lo0 (i32x4.mul (local.get $bo0) (local.get $m0)))
      (local.set $lo1 (i32x4.mul (local.get $bo2) (local.get $m1)))
      (local.set $hi0 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $bo0) (local.get $m0))
        (i64x2.extmul_high_i32x4_u (local.get $bo0) (local.get $m0))))
      (local.set $hi1 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $bo2) (local.get $m1))
        (i64x2.extmul_high_i32x4_u (local.get $bo2) (local.get $m1))))
      (local.set $n0 (v128.xor (local.get $bo1) (v128.xor (local.get $hi1) (local.get $lo1))))
      (local.set $n1 (v128.xor (local.get $bh2) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo1) (local.get $lo1))))
      (local.set $n2 (v128.xor (local.get $bo3) (v128.xor (local.get $hi0) (local.get $lo0))))
      (local.set $n3 (v128.xor (local.get $bh3) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo0) (local.get $lo0))))
      (local.set $bh0 (v128.xor (local.get $bh0)
        (v128.or (i32x4.shl (local.get $bh1) (i32.const 7)) (i32x4.shr_u (local.get $bh1) (i32.const 25)))))
      (local.set $bh1 (v128.xor (local.get $bh1)
        (v128.or (i32x4.shl (local.get $bh2) (i32.const 13)) (i32x4.shr_u (local.get $bh2) (i32.const 19)))))
      (local.set $bh2 (v128.xor (local.get $bh2)
        (v128.or (i32x4.shl (local.get $bh3) (i32.const 22)) (i32x4.shr_u (local.get $bh3) (i32.const 10)))))
      (local.set $bh3 (v128.xor (local.get $bh3)
        (v128.or (i32x4.shl (local.get $bh0) (i32.const 3)) (i32x4.shr_u (local.get $bh0) (i32.const 29)))))
      (local.set $bo0 (v128.xor (i32x4.add (local.get $bh0) (local.get $weyl)) (local.get $n0)))
      (local.set $bh0 (v128.xor (local.get $n0) (local.get $rc)))
      (local.set $bo1 (local.get $bh1))
      (local.set $bo2 (local.get $bh2))
      (local.set $bo3 (local.get $bh3))
      (local.set $bh1 (local.get $n1))
      (local.set $bh2 (local.get $n2))
      (local.set $bh3 (local.get $n3))
      (br_if $round (i32.lt_u (local.tee $r (i32.add (local.get $r) (i32.const 1))) (i32.const 8))))
    (v128.store offset=0 (i32.const 0) (local.get $ao0))
    (v128.store offset=16 (i32.const 0) (local.get $bo0))
    (v128.store offset=32 (i32.const 0) (local.get $ao1))
    (v128.store offset=48 (i32.const 0) (local.get $bo1))
    (v128.store offset=64 (i32.const 0) (local.get $ao2))
    (v128.store offset=80 (i32.const 0) (local.get $bo2))
    (v128.store offset=96 (i32.const 0) (local.get $ao3))
    (v128.store offset=112 (i32.const 0) (local.get $bo3))
    (v128.store offset=128 (i32.const 0) (local.get $ah0))
    (v128.store offset=144 (i32.const 0) (local.get $bh0))
    (v128.store offset=160 (i32.const 0) (local.get $ah1))
    (v128.store offset=176 (i32.const 0) (local.get $bh1))
    (v128.store offset=192 (i32.const 0) (local.get $ah2))
    (v128.store offset=208 (i32.const 0) (local.get $bh2))
    (v128.store offset=224 (i32.const 0) (local.get $ah3))
    (v128.store offset=240 (i32.const 0) (local.get $bh3))
    (i32.store offset=280 (i32.const 0) (i32.const 0)))

  ;; rows(skip, n): step the state skip times, then n times writing row r at 512 + 128 r, with
  ;; n at most the capacity. A step at row K of a group seeds the next group first.
  (func (export "rows") (param $skip i32) (param $n i32)
    (local $r i32) (local $end i32) (local $at i32) (local $j i32) (local $c i32)
    (local $one v128) (local $weyl v128)
    (local $m0 v128) (local $m1 v128) (local $lo0 v128) (local $lo1 v128)
    (local $hi0 v128) (local $hi1 v128) (local $n0 v128) (local $n1 v128) (local $n2 v128)
    (local $n3 v128) (local $t0 v128) (local $t1 v128) (local $t2 v128) (local $t3 v128)
    (local $ao0 v128) (local $ao1 v128) (local $ao2 v128) (local $ao3 v128)
    (local $ah0 v128) (local $ah1 v128) (local $ah2 v128) (local $ah3 v128)
    (local $bo0 v128) (local $bo1 v128) (local $bo2 v128) (local $bo3 v128)
    (local $bh0 v128) (local $bh1 v128) (local $bh2 v128) (local $bh3 v128)
    (local.set $end (i32.add (local.get $skip) (local.get $n)))
    (if (i32.eqz (local.get $end)) (then (return)))
    (local.set $one (i32x4.splat (i32.const 1)))
    (local.set $weyl (i32x4.splat (i32.const 0x9e3779b9)))
    (local.set $j (i32.load offset=280 (i32.const 0)))
    ;; The first row lands at 512 - 128 skip, so row r of the output is at 512 + 128 (r - skip).
    (local.set $at (i32.sub (i32.const 512) (i32.shl (local.get $skip) (i32.const 7))))
    (loop $group
    (block $next
    (local.set $ao0 (v128.load offset=0 (i32.const 0)))
    (local.set $bo0 (v128.load offset=16 (i32.const 0)))
    (local.set $ao1 (v128.load offset=32 (i32.const 0)))
    (local.set $bo1 (v128.load offset=48 (i32.const 0)))
    (local.set $ao2 (v128.load offset=64 (i32.const 0)))
    (local.set $bo2 (v128.load offset=80 (i32.const 0)))
    (local.set $ao3 (v128.load offset=96 (i32.const 0)))
    (local.set $bo3 (v128.load offset=112 (i32.const 0)))
    (local.set $ah0 (v128.load offset=128 (i32.const 0)))
    (local.set $bh0 (v128.load offset=144 (i32.const 0)))
    (local.set $ah1 (v128.load offset=160 (i32.const 0)))
    (local.set $bh1 (v128.load offset=176 (i32.const 0)))
    (local.set $ah2 (v128.load offset=192 (i32.const 0)))
    (local.set $bh2 (v128.load offset=208 (i32.const 0)))
    (local.set $ah3 (v128.load offset=224 (i32.const 0)))
    (local.set $bh3 (v128.load offset=240 (i32.const 0)))
    (loop $step
      (br_if $next (i32.eq (local.get $j) (i32.load offset=284 (i32.const 0))))
      ;; T on half a
      (local.set $m0 (v128.or (local.get $ah0) (local.get $one)))
      (local.set $m1 (v128.or (local.get $ah1) (local.get $one)))
      (local.set $lo0 (i32x4.mul (local.get $ao0) (local.get $m0)))
      (local.set $lo1 (i32x4.mul (local.get $ao2) (local.get $m1)))
      (local.set $hi0 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $ao0) (local.get $m0))
        (i64x2.extmul_high_i32x4_u (local.get $ao0) (local.get $m0))))
      (local.set $hi1 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $ao2) (local.get $m1))
        (i64x2.extmul_high_i32x4_u (local.get $ao2) (local.get $m1))))
      (local.set $n0 (v128.xor (local.get $ao1) (v128.xor (local.get $hi1) (local.get $lo1))))
      (local.set $ao1 (v128.xor (local.get $ah2) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo1) (local.get $lo1))))
      (local.set $ao2 (v128.xor (local.get $ao3) (v128.xor (local.get $hi0) (local.get $lo0))))
      (local.set $ao3 (v128.xor (local.get $ah3) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo0) (local.get $lo0))))
      (local.set $ao0 (local.get $n0))
      (local.set $ah0 (v128.xor (local.get $ah0)
        (v128.or (i32x4.shl (local.get $ah1) (i32.const 7)) (i32x4.shr_u (local.get $ah1) (i32.const 25)))))
      (local.set $ah1 (v128.xor (local.get $ah1)
        (v128.or (i32x4.shl (local.get $ah2) (i32.const 13)) (i32x4.shr_u (local.get $ah2) (i32.const 19)))))
      (local.set $ah2 (v128.xor (local.get $ah2)
        (v128.or (i32x4.shl (local.get $ah3) (i32.const 22)) (i32x4.shr_u (local.get $ah3) (i32.const 10)))))
      (local.set $ah3 (v128.xor (local.get $ah3)
        (v128.or (i32x4.shl (local.get $ah0) (i32.const 3)) (i32x4.shr_u (local.get $ah0) (i32.const 29)))))
      (local.set $ah0 (v128.xor (i32x4.add (local.get $ah0) (local.get $weyl)) (local.get $n0)))
      ;; T on half b
      (local.set $m0 (v128.or (local.get $bh0) (local.get $one)))
      (local.set $m1 (v128.or (local.get $bh1) (local.get $one)))
      (local.set $lo0 (i32x4.mul (local.get $bo0) (local.get $m0)))
      (local.set $lo1 (i32x4.mul (local.get $bo2) (local.get $m1)))
      (local.set $hi0 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $bo0) (local.get $m0))
        (i64x2.extmul_high_i32x4_u (local.get $bo0) (local.get $m0))))
      (local.set $hi1 (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
        (i64x2.extmul_low_i32x4_u (local.get $bo2) (local.get $m1))
        (i64x2.extmul_high_i32x4_u (local.get $bo2) (local.get $m1))))
      (local.set $n0 (v128.xor (local.get $bo1) (v128.xor (local.get $hi1) (local.get $lo1))))
      (local.set $bo1 (v128.xor (local.get $bh2) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo1) (local.get $lo1))))
      (local.set $bo2 (v128.xor (local.get $bo3) (v128.xor (local.get $hi0) (local.get $lo0))))
      (local.set $bo3 (v128.xor (local.get $bh3) (i8x16.shuffle 2 3 0 1 6 7 4 5 10 11 8 9 14 15 12 13
        (local.get $lo0) (local.get $lo0))))
      (local.set $bo0 (local.get $n0))
      (local.set $bh0 (v128.xor (local.get $bh0)
        (v128.or (i32x4.shl (local.get $bh1) (i32.const 7)) (i32x4.shr_u (local.get $bh1) (i32.const 25)))))
      (local.set $bh1 (v128.xor (local.get $bh1)
        (v128.or (i32x4.shl (local.get $bh2) (i32.const 13)) (i32x4.shr_u (local.get $bh2) (i32.const 19)))))
      (local.set $bh2 (v128.xor (local.get $bh2)
        (v128.or (i32x4.shl (local.get $bh3) (i32.const 22)) (i32x4.shr_u (local.get $bh3) (i32.const 10)))))
      (local.set $bh3 (v128.xor (local.get $bh3)
        (v128.or (i32x4.shl (local.get $bh0) (i32.const 3)) (i32x4.shr_u (local.get $bh0) (i32.const 29)))))
      (local.set $bh0 (v128.xor (i32x4.add (local.get $bh0) (local.get $weyl)) (local.get $n0)))
      (if (i32.ge_u (local.get $r) (local.get $skip))
        (then
          ;; Lane l's block is (o0, o1, o2, o3)[l]: a 4 x 4 transpose per half.
          (local.set $t0 (i8x16.shuffle 0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 (local.get $ao0) (local.get $ao1)))
          (local.set $t1 (i8x16.shuffle 0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 (local.get $ao2) (local.get $ao3)))
          (local.set $t2 (i8x16.shuffle 8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31 (local.get $ao0) (local.get $ao1)))
          (local.set $t3 (i8x16.shuffle 8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31 (local.get $ao2) (local.get $ao3)))
          (v128.store offset=0 (local.get $at)
            (i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 (local.get $t0) (local.get $t1)))
          (v128.store offset=16 (local.get $at)
            (i8x16.shuffle 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31 (local.get $t0) (local.get $t1)))
          (v128.store offset=32 (local.get $at)
            (i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 (local.get $t2) (local.get $t3)))
          (v128.store offset=48 (local.get $at)
            (i8x16.shuffle 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31 (local.get $t2) (local.get $t3)))
          (local.set $t0 (i8x16.shuffle 0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 (local.get $bo0) (local.get $bo1)))
          (local.set $t1 (i8x16.shuffle 0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 (local.get $bo2) (local.get $bo3)))
          (local.set $t2 (i8x16.shuffle 8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31 (local.get $bo0) (local.get $bo1)))
          (local.set $t3 (i8x16.shuffle 8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31 (local.get $bo2) (local.get $bo3)))
          (v128.store offset=64 (local.get $at)
            (i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 (local.get $t0) (local.get $t1)))
          (v128.store offset=80 (local.get $at)
            (i8x16.shuffle 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31 (local.get $t0) (local.get $t1)))
          (v128.store offset=96 (local.get $at)
            (i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 (local.get $t2) (local.get $t3)))
          (v128.store offset=112 (local.get $at)
            (i8x16.shuffle 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31 (local.get $t2) (local.get $t3)))))
      (local.set $at (i32.add (local.get $at) (i32.const 128)))
      (local.set $j (i32.add (local.get $j) (i32.const 1)))
      (br_if $step (i32.lt_u (local.tee $r (i32.add (local.get $r) (i32.const 1))) (local.get $end))))
    (v128.store offset=0 (i32.const 0) (local.get $ao0))
    (v128.store offset=16 (i32.const 0) (local.get $bo0))
    (v128.store offset=32 (i32.const 0) (local.get $ao1))
    (v128.store offset=48 (i32.const 0) (local.get $bo1))
    (v128.store offset=64 (i32.const 0) (local.get $ao2))
    (v128.store offset=80 (i32.const 0) (local.get $bo2))
    (v128.store offset=96 (i32.const 0) (local.get $ao3))
    (v128.store offset=112 (i32.const 0) (local.get $bo3))
    (v128.store offset=128 (i32.const 0) (local.get $ah0))
    (v128.store offset=144 (i32.const 0) (local.get $bh0))
    (v128.store offset=160 (i32.const 0) (local.get $ah1))
    (v128.store offset=176 (i32.const 0) (local.get $bh1))
    (v128.store offset=192 (i32.const 0) (local.get $ah2))
    (v128.store offset=208 (i32.const 0) (local.get $bh2))
    (v128.store offset=224 (i32.const 0) (local.get $ah3))
    (v128.store offset=240 (i32.const 0) (local.get $bh3))
    (i32.store offset=280 (i32.const 0) (local.get $j))
    (return))
    ;; The group is used up: the next one starts at chunk c + 8, with the carry into the high word.
    (local.set $c (i32.add (i32.load offset=272 (i32.const 0)) (i32.const 8)))
    (i32.store offset=272 (i32.const 0) (local.get $c))
    (if (i32.lt_u (local.get $c) (i32.const 8))
      (then (i32.store offset=276 (i32.const 0)
        (i32.add (i32.load offset=276 (i32.const 0)) (i32.const 1)))))
    (call $seed)
    (local.set $j (i32.const 0))
    (br $group)))

  ;; exp64(n): z = -ln(1 - z) for the n doubles at 33280, n even and at most 2048, two lanes at
  ;; a time with the arithmetic of derived.ts neg2Log64, which also writes the polynomial, its
  ;; limits and the other constants to 49664 before the first call: c[k] at 8 k, lim[k] at
  ;; 64 + 8 k, then SPLIT, LN2_HI, LN2_LO, LN2_LO_H, LN2_LO_L and LAST_LIM from 128. The plain
  ;; Horner chain runs first with a per-lane bound on each step, as hornerLimits of derived.ts
  ;; but from the exponents of the sum and the product. A pair with a lane that failed a step
  ;; takes the chain again with Dekker's product and the fused sum on every step, which gives
  ;; the same value on a lane that passed. The last step adds nk LN2_LO with the same kind of
  ;; bound. The lane conditions of the fused sums are bit selects.
  (global (export "exp") i32 (i32.const 33280))
  (global (export "exp_capacity") i32 (i32.const 2048))
  (global (export "consts") i32 (i32.const 49664))
  (func (export "exp64") (param $n i32)
    (local $i i32) (local $k i32) (local $zero v128) (local $one v128) (local $split v128)
    (local $x v128) (local $nk v128) (local $m v128) (local $s v128) (local $w v128)
    (local $a v128) (local $c v128) (local $p v128) (local $sum v128) (local $t v128)
    (local $e v128) (local $v v128) (local $z v128) (local $r v128) (local $r2 v128)
    (local $h v128) (local $l v128) (local $bh v128) (local $bl v128) (local $bv v128)
    (local $ev v128) (local $y v128) (local $q v128) (local $bb v128) (local $bz v128)
    (local $bad v128)
    (local.set $zero (f64x2.splat (f64.const 0)))
    (local.set $one (f64x2.splat (f64.const 1)))
    (local.set $split (v128.load64_splat offset=49792 (i32.const 0)))
    (local.set $i (i32.const 33280))
    (local.set $n (i32.add (local.get $i) (i32.shl (local.get $n) (i32.const 3))))
    (block $done (loop $next
      (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
      (local.set $x (f64x2.sub (local.get $one) (v128.load (local.get $i))))
      ;; ix = bits + 0x95f62 2^32, nk = 1023 - (ix >> 52) as a double through 2^52, and the
      ;; mantissa from the low 52 bits of ix under the exponent of sqrt(1/2).
      (local.set $x (i64x2.add (local.get $x)
        (v128.const i64x2 0x00095f6200000000 0x00095f6200000000)))
      (local.set $nk (f64x2.sub (f64x2.splat (f64.const 1023)) (f64x2.sub
        (v128.or (i64x2.shr_u (local.get $x) (i32.const 52))
          (v128.const i64x2 0x4330000000000000 0x4330000000000000))
        (f64x2.splat (f64.const 0x1p52)))))
      (local.set $m (i64x2.add
        (v128.and (local.get $x) (v128.const i64x2 0x000fffffffffffff 0x000fffffffffffff))
        (v128.const i64x2 0x3fe6a09e00000000 0x3fe6a09e00000000)))
      (local.set $s (f64x2.div (f64x2.sub (local.get $m) (local.get $one))
        (f64x2.add (local.get $m) (local.get $one))))
      (local.set $w (f64x2.mul (local.get $s) (local.get $s)))
      ;; The plain chain first, with the lanes that failed a limit collected in $bad. Only a pair
      ;; with such a lane takes the checked chain below, since a branch per step costs about
      ;; as much as the arithmetic.
      (local.set $bad (local.get $zero))
      (local.set $a (v128.load64_splat offset=49664 (i32.const 0)))
      (local.set $k (i32.const 8))
      (loop $plain
        (local.set $c (v128.load64_splat offset=49664 (local.get $k)))
        (local.set $p (f64x2.mul (local.get $w) (local.get $a)))
        (local.set $sum (f64x2.add (local.get $c) (local.get $p)))
        (local.set $t (f64x2.sub (local.get $p) (f64x2.sub (local.get $sum) (local.get $c))))
        ;; The lane fails when |t| reaches half the spacing of sum less half the spacing of p,
        ;; both from the exponent bits, since the error of p is at most the latter.
        (local.set $bad (v128.or (local.get $bad) (f64x2.ge (f64x2.abs (local.get $t))
          (f64x2.mul (f64x2.sub
            (v128.and (local.get $sum) (v128.const i64x2 0x7ff0000000000000 0x7ff0000000000000))
            (v128.and (local.get $p) (v128.const i64x2 0x7ff0000000000000 0x7ff0000000000000)))
            (f64x2.splat (f64.const 0x1p-53))))))
        (local.set $a (local.get $sum))
        (local.set $k (i32.add (local.get $k) (i32.const 8)))
        (br_if $plain (i32.lt_u (local.get $k) (i32.const 56))))
      (if (v128.any_true (local.get $bad)) (then
      (local.set $a (v128.load64_splat offset=49664 (i32.const 0)))
      (local.set $k (i32.const 8))
      (loop $step
        (local.set $c (v128.load64_splat offset=49664 (local.get $k)))
        (local.set $p (f64x2.mul (local.get $w) (local.get $a)))
        (local.set $sum (f64x2.add (local.get $c) (local.get $p)))
        (local.set $t (f64x2.sub (local.get $p) (f64x2.sub (local.get $sum) (local.get $c))))
        (block
            (local.set $h (f64x2.mul (local.get $split) (local.get $w)))
            (local.set $h (f64x2.sub (local.get $h) (f64x2.sub (local.get $h) (local.get $w))))
            (local.set $l (f64x2.sub (local.get $w) (local.get $h)))
            (local.set $bh (f64x2.mul (local.get $split) (local.get $a)))
            (local.set $bh (f64x2.sub (local.get $bh) (f64x2.sub (local.get $bh) (local.get $a))))
            (local.set $bl (f64x2.sub (local.get $a) (local.get $bh)))
            (local.set $e (f64x2.add (f64x2.add (f64x2.add
              (f64x2.sub (f64x2.mul (local.get $h) (local.get $bh)) (local.get $p))
              (f64x2.mul (local.get $h) (local.get $bl)))
              (f64x2.mul (local.get $l) (local.get $bh)))
              (f64x2.mul (local.get $l) (local.get $bl))))
            (local.set $v (f64x2.add (local.get $t) (local.get $e)))
            (local.set $z (f64x2.add (local.get $sum) (local.get $v)))
            (local.set $r (f64x2.sub (local.get $v) (f64x2.sub (local.get $z) (local.get $sum))))
            (local.set $r2 (f64x2.add (local.get $r) (local.get $r)))
            (local.set $bv (f64x2.sub (local.get $v) (local.get $t)))
            (local.set $ev (f64x2.add
              (f64x2.sub (local.get $t) (f64x2.sub (local.get $v) (local.get $bv)))
              (f64x2.sub (local.get $e) (local.get $bv))))
            (local.set $a (v128.bitselect
              (f64x2.add (local.get $z) (local.get $r2)) (local.get $z)
              (v128.and (v128.and
                (f64x2.ne (local.get $r) (local.get $zero))
                (f64x2.eq (f64x2.sub (f64x2.add (local.get $z) (local.get $r2)) (local.get $z))
                  (local.get $r2)))
              (v128.and
                (f64x2.ne (local.get $ev) (local.get $zero))
                (v128.not (v128.xor (f64x2.gt (local.get $ev) (local.get $zero))
                  (f64x2.gt (local.get $r) (local.get $zero)))))))))
        (local.set $k (i32.add (local.get $k) (i32.const 8)))
        (br_if $step (i32.lt_u (local.get $k) (i32.const 56))))))
      ;; y = nk LN2_HI - 4 s p, exact in the product, then y + nk LN2_LO fused.
      (local.set $y (f64x2.add
        (f64x2.mul (local.get $nk) (v128.load64_splat offset=49800 (i32.const 0)))
        (f64x2.mul (f64x2.mul (local.get $s) (f64x2.splat (f64.const -4))) (local.get $a))))
      (local.set $q (f64x2.mul (local.get $nk) (v128.load64_splat offset=49808 (i32.const 0))))
      (local.set $z (f64x2.add (local.get $y) (local.get $q)))
      (local.set $t (f64x2.sub (local.get $q) (f64x2.sub (local.get $z) (local.get $y))))
      ;; Half the spacing of y from its exponent bits, less the bound 2^-74 on the error of q.
      (if (i32.eqz (i64x2.all_true (f64x2.lt (f64x2.abs (local.get $t))
            (f64x2.sub (f64x2.mul
              (v128.and (local.get $y) (v128.const i64x2 0x7ff0000000000000 0x7ff0000000000000))
              (f64x2.splat (f64.const 0x1p-53)))
              (f64x2.splat (f64.const 0x1p-74))))))
        (then
          (local.set $e (f64x2.add
            (f64x2.sub (f64x2.mul (local.get $nk) (v128.load64_splat offset=49816 (i32.const 0)))
              (local.get $q))
            (f64x2.mul (local.get $nk) (v128.load64_splat offset=49824 (i32.const 0)))))
          (local.set $bb (f64x2.sub (local.get $z) (local.get $y)))
          (local.set $t (f64x2.add
            (f64x2.sub (local.get $y) (f64x2.sub (local.get $z) (local.get $bb)))
            (f64x2.sub (local.get $q) (local.get $bb))))
          (local.set $v (f64x2.add (local.get $t) (local.get $e)))
          (local.set $bv (f64x2.sub (local.get $v) (local.get $t)))
          (local.set $ev (f64x2.add
            (f64x2.sub (local.get $t) (f64x2.sub (local.get $v) (local.get $bv)))
            (f64x2.sub (local.get $e) (local.get $bv))))
          (local.set $sum (f64x2.add (local.get $z) (local.get $v)))
          (local.set $bz (f64x2.sub (local.get $sum) (local.get $z)))
          (local.set $r (f64x2.add
            (f64x2.sub (local.get $z) (f64x2.sub (local.get $sum) (local.get $bz)))
            (f64x2.sub (local.get $v) (local.get $bz))))
          (local.set $r2 (f64x2.add (local.get $r) (local.get $r)))
          (local.set $z (v128.bitselect
            (f64x2.add (local.get $sum) (local.get $r2)) (local.get $sum)
            (v128.and (v128.and
              (f64x2.ne (local.get $ev) (local.get $zero))
              (f64x2.ne (local.get $r) (local.get $zero)))
            (v128.and
              (f64x2.eq (f64x2.sub (f64x2.add (local.get $sum) (local.get $r2)) (local.get $sum))
                (local.get $r2))
              (v128.not (v128.xor (f64x2.gt (local.get $ev) (local.get $zero))
                (f64x2.gt (local.get $r) (local.get $zero))))))))))
      (v128.store (local.get $i) (f64x2.mul (local.get $z) (f64x2.splat (f64.const 0.5))))
      (local.set $i (i32.add (local.get $i) (i32.const 16)))
      (br $next))))
)
