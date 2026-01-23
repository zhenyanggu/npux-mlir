#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "act_test_quant_sym"} {
  func.func @main_graph(%arg0: memref<1x32xf32> {onnx.name = "input"}) -> (memref<1x32xf32> {onnx.name = "output"}) attributes {llvm.emit_c_interface} {
    %0 = "krnl.global"() {name = "constant_0", shape = [32], value = dense<[-9.82335652E-4, -0.0923864468, 0.0308713373, 0.125592053, -0.0481653772, 0.0492569357, -0.0528857373, 0.191135094, -0.102464393, 0.0097812349, -0.0817779526, 0.00841632765, -0.0643123686, -0.125691578, -0.048106797, -0.115775935, -0.125293836, 7.638620e-02, 0.0334553234, -0.0247902982, 0.0771105587, 0.103332363, -0.129471868, 0.00529868389, 0.0113818403, -0.0332146063, 0.0631280169, 0.0246195775, -0.123717859, -0.0606194735, 0.0602609515, -0.0785476937]> : tensor<32xf32>} : () -> memref<32xf32>
    %1 = "krnl.global"() {name = "constant_1", shape = [32], value = dense<[-0.174824119, 0.0794920623, -0.0248840358, -0.109353125, -0.0772407278, -0.0361809246, 0.219115898, 0.187830746, 0.245069295, -0.0233103242, 0.0372669809, -0.0510918386, 0.044748459, -0.00205091923, -0.0826271325, -0.157617912, 0.100945644, 0.106719315, 0.0168068167, -0.0849939808, -0.0280704256, -0.0105170989, 0.18064329, -0.0139956269, -0.035048712, 0.0254881084, -0.0581126623, -0.12114203, 0.0639609322, -0.0395348035, -0.140444294, -0.0503093414]> : tensor<32xf32>} : () -> memref<32xf32>
    %2 = "krnl.global"() {name = "constant_2", shape = [], value = dense<0> : tensor<i8>} : () -> memref<i8>
    %3 = "krnl.global"() {name = "constant_3", shape = [], value = dense<0.0227836408> : tensor<f32>} : () -> memref<f32>
    %4 = "krnl.global"() {name = "constant_4", shape = [], value = dense<0.00787401571> : tensor<f32>} : () -> memref<f32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x32xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %c1 step %c1_0 {
      %c0_6 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1_7 = arith.constant 1 : index
      scf.for %arg2 = %c0_6 to %c32 step %c1_7 {
        %c0_8 = arith.constant 0 : index
        %5 = memref.load %arg0[%c0_8, %arg2] : memref<1x32xf32>
        %6 = memref.load %0[%arg2] : memref<32xf32>
        %7 = arith.addf %5, %6 : f32
        memref.store %7, %alloc[%arg1, %arg2] : memref<1x32xf32>
      }
    }
    %alloc_1 = memref.alloc() : memref<1x32xf32>
    call @npu_kernel_0(%alloc, %3, %2, %4, %alloc_1) : (memref<1x32xf32>, memref<f32>, memref<i8>, memref<f32>, memref<1x32xf32>) -> ()
    %alloc_2 = memref.alloc() {alignment = 16 : i64} : memref<1x32xf32>
    %c0_3 = arith.constant 0 : index
    %c1_4 = arith.constant 1 : index
    %c1_5 = arith.constant 1 : index
    scf.for %arg1 = %c0_3 to %c1_4 step %c1_5 {
      %c0_6 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1_7 = arith.constant 1 : index
      scf.for %arg2 = %c0_6 to %c32 step %c1_7 {
        %c0_8 = arith.constant 0 : index
        %5 = memref.load %alloc_1[%c0_8, %arg2] : memref<1x32xf32>
        %6 = memref.load %1[%arg2] : memref<32xf32>
        %7 = arith.addf %5, %6 : f32
        memref.store %7, %alloc_2[%arg1, %arg2] : memref<1x32xf32>
      }
    }
    return %alloc_2 : memref<1x32xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22output\22 }\0A\0A]\00"} : () -> ()
  func.func private @npu_kernel_0(%arg0: memref<1x32xf32>, %arg1: memref<f32>, %arg2: memref<i8>, %arg3: memref<f32>, %arg4: memref<1x32xf32>) attributes {llvm.emit_c_interface, npu.target = "npu"} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant -1.280000e+02 : f32
    %cst_3 = arith.constant 1.270000e+02 : f32
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x32xi8>
    %0 = memref.load %arg1[] : memref<f32>
    %1 = memref.load %arg2[] : memref<i8>
    %2 = arith.extsi %1 : i8 to i32
    %3 = arith.sitofp %2 : i32 to f32
    %alloc_4 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
    %c0 = arith.constant 0 : index
    memref.store %c32, %alloc_4[%c0] : memref<1xindex>
    %reshape = memref.reshape %arg0(%alloc_4) : (memref<1x32xf32>, memref<1xindex>) -> memref<32xf32>
    %alloc_5 = memref.alloc() {alignment = 16 : i64} : memref<1xindex>
    %c0_6 = arith.constant 0 : index
    memref.store %c32, %alloc_5[%c0_6] : memref<1xindex>
    %reshape_7 = memref.reshape %alloc(%alloc_5) : (memref<1x32xi8>, memref<1xindex>) -> memref<32xi8>
    %c0_8 = arith.constant 0 : index
    %c32_9 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0_8 to %c32_9 step %c1 {
      %4 = memref.load %reshape[%arg5] : memref<32xf32>
      %5 = arith.divf %4, %0 : f32
      %6 = math.floor %5 : f32
      %7 = arith.subf %5, %6 : f32
      %8 = arith.cmpf ogt, %7, %cst : f32
      %9 = arith.addf %6, %cst_1 : f32
      %10 = arith.select %8, %9, %6 : f32
      %11 = arith.mulf %6, %cst : f32
      %12 = math.floor %11 : f32
      %13 = arith.mulf %12, %cst_0 : f32
      %14 = arith.subf %6, %13 : f32
      %15 = arith.cmpf oeq, %14, %cst_1 : f32
      %16 = arith.select %15, %9, %6 : f32
      %17 = arith.cmpf oeq, %7, %cst : f32
      %18 = arith.select %17, %16, %10 : f32
      %19 = arith.addf %18, %3 : f32
      %20 = arith.maxnumf %19, %cst_2 : f32
      %21 = arith.minnumf %20, %cst_3 : f32
      %22 = arith.fptosi %21 : f32 to i32
      %23 = arith.trunci %22 : i32 to i8
      memref.store %23, %reshape_7[%arg5] : memref<32xi8>
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi8>
    %alloc_11 = memref.alloc() : memref<1x32xi8, 2>
    memref.copy %alloc, %alloc_11 : memref<1x32xi8> to memref<1x32xi8, 2>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], library_call = "npu_layernorm"} ins(%alloc_11 : memref<1x32xi8, 2>) outs(%alloc_11 : memref<1x32xi8, 2>) attrs =  {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, in_scale = 0.0227836408 : f32, in_zp = 0 : i32, npu.target = "npu", out_scale = 0.0231352877 : f32, out_zp = 0 : i32} {
    ^bb0(%in: i8, %out: i8):
      %4 = arith.addi %in, %in : i8
      linalg.yield %4 : i8
    }
    memref.copy %alloc_11, %alloc_10 : memref<1x32xi8, 2> to memref<1x32xi8>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi8>
    %alloc_13 = memref.alloc() : memref<1x32xi8, 2>
    memref.copy %alloc_10, %alloc_13 : memref<1x32xi8> to memref<1x32xi8, 2>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], library_call = "npu_gelu"} ins(%alloc_13 : memref<1x32xi8, 2>) outs(%alloc_13 : memref<1x32xi8, 2>) attrs =  {in_scale = 0.0231352877 : f32, in_zp = 0 : i32, npu.target = "npu", npu.tiled, npu.trivial_tiling, out_scale = 0.0230970979 : f32, out_zp = 0 : i16} {
    ^bb0(%in: i8, %out: i8):
      %4 = arith.addi %in, %in : i8
      linalg.yield %4 : i8
    }
    memref.copy %alloc_13, %alloc_12 : memref<1x32xi8, 2> to memref<1x32xi8>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32xi8>
    %alloc_15 = memref.alloc() : memref<1x32xi8, 2>
    memref.copy %alloc_12, %alloc_15 : memref<1x32xi8> to memref<1x32xi8, 2>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], library_call = "npu_softmax"} ins(%alloc_15 : memref<1x32xi8, 2>) outs(%alloc_15 : memref<1x32xi8, 2>) attrs =  {axis = -1 : i64, in_scale = 0.0230970979 : f32, in_zp = 0 : i32, npu.target = "npu", out_scale = 0.00787401571 : f32, out_zp = 0 : i16} {
    ^bb0(%in: i8, %out: i8):
      %4 = arith.addi %in, %in : i8
      linalg.yield %4 : i8
    }
    memref.copy %alloc_15, %alloc_14 : memref<1x32xi8, 2> to memref<1x32xi8>
    %c0_16 = arith.constant 0 : index
    %c1_17 = arith.constant 1 : index
    %c1_18 = arith.constant 1 : index
    scf.for %arg5 = %c0_16 to %c1_17 step %c1_18 {
      %c0_19 = arith.constant 0 : index
      %c32_20 = arith.constant 32 : index
      %c1_21 = arith.constant 1 : index
      scf.for %arg6 = %c0_19 to %c32_20 step %c1_21 {
        %4 = memref.load %alloc_14[%arg5, %arg6] : memref<1x32xi8>
        %5 = memref.load %arg3[] : memref<f32>
        %6 = memref.load %arg2[] : memref<i8>
        %7 = arith.extsi %4 : i8 to i32
        %8 = arith.sitofp %7 : i32 to f32
        %9 = arith.extsi %6 : i8 to i32
        %10 = arith.sitofp %9 : i32 to f32
        %11 = arith.subf %8, %10 : f32
        %12 = arith.mulf %11, %5 : f32
        memref.store %12, %arg4[%arg5, %arg6] : memref<1x32xf32>
      }
    }
    return
  }
}

