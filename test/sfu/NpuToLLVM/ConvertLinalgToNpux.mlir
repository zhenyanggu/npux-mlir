module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "act_test_quant_sym"} {
  func.func @main_graph(%arg0: memref<1x32xf32> {onnx.name = "input"}) -> (memref<1x32xf32> {onnx.name = "output"}) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = "npux.init"() : () -> i32
    %1 = "krnl.global"() {name = "constant_0", shape = [32], value = dense<[-9.82335652E-4, -0.0923864468, 0.0308713373, 0.125592053, -0.0481653772, 0.0492569357, -0.0528857373, 0.191135094, -0.102464393, 0.0097812349, -0.0817779526, 0.00841632765, -0.0643123686, -0.125691578, -0.048106797, -0.115775935, -0.125293836, 7.638620e-02, 0.0334553234, -0.0247902982, 0.0771105587, 0.103332363, -0.129471868, 0.00529868389, 0.0113818403, -0.0332146063, 0.0631280169, 0.0246195775, -0.123717859, -0.0606194735, 0.0602609515, -0.0785476937]> : tensor<32xf32>} : () -> memref<32xf32>
    %2 = "krnl.global"() {name = "constant_1", shape = [32], value = dense<[-0.174824119, 0.0794920623, -0.0248840358, -0.109353125, -0.0772407278, -0.0361809246, 0.219115898, 0.187830746, 0.245069295, -0.0233103242, 0.0372669809, -0.0510918386, 0.044748459, -0.00205091923, -0.0826271325, -0.157617912, 0.100945644, 0.106719315, 0.0168068167, -0.0849939808, -0.0280704256, -0.0105170989, 0.18064329, -0.0139956269, -0.035048712, 0.0254881084, -0.0581126623, -0.12114203, 0.0639609322, -0.0395348035, -0.140444294, -0.0503093414]> : tensor<32xf32>} : () -> memref<32xf32>
    %3 = "krnl.global"() {name = "constant_2", shape = [], value = dense<0> : tensor<i8>} : () -> memref<i8>
    %4 = "krnl.global"() {name = "constant_3", shape = [], value = dense<0.0227836408> : tensor<f32>} : () -> memref<f32>
    %5 = "krnl.global"() {name = "constant_4", shape = [], value = dense<0.00787401571> : tensor<f32>} : () -> memref<f32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x32xf32>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %6 = memref.load %arg0[%c0, %arg1] : memref<1x32xf32>
      %7 = memref.load %1[%arg1] : memref<32xf32>
      %8 = arith.addf %6, %7 : f32
      memref.store %8, %alloc[%c0, %arg1] : memref<1x32xf32>
    }
    %alloc_0 = memref.alloc() : memref<1x32xf32>
    call @npu_kernel_0(%alloc, %4, %3, %5, %alloc_0) : (memref<1x32xf32>, memref<f32>, memref<i8>, memref<f32>, memref<1x32xf32>) -> ()
    memref.dealloc %alloc : memref<1x32xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x32xf32>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %6 = memref.load %alloc_0[%c0, %arg1] : memref<1x32xf32>
      %7 = memref.load %2[%arg1] : memref<32xf32>
      %8 = arith.addf %6, %7 : f32
      memref.store %8, %alloc_1[%c0, %arg1] : memref<1x32xf32>
    }
    memref.dealloc %alloc_0 : memref<1x32xf32>
    "npux.destroy"() : () -> ()
    return %alloc_1 : memref<1x32xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22output\22 }\0A\0A]\00"} : () -> ()
  func.func private @npu_kernel_0(%arg0: memref<1x32xf32>, %arg1: memref<f32>, %arg2: memref<i8>, %arg3: memref<f32>, %arg4: memref<1x32xf32>) attributes {llvm.emit_c_interface, npu.target = "npu"} {
    %c-8_i16 = arith.constant -8 : i16
    %c32512_i16 = arith.constant 32512 : i16
    %c24219_i16 = arith.constant 24219 : i16
    %c22167_i16 = arith.constant 22167 : i16
    %c24259_i16 = arith.constant 24259 : i16
    %true = arith.constant true
    %c8_i8 = arith.constant 8 : i8
    %c-9_i16 = arith.constant -9 : i16
    %c22131_i16 = arith.constant 22131 : i16
    %c-20_i16 = arith.constant -20 : i16
    %c23890_i16 = arith.constant 23890 : i16
    %c1_i8 = arith.constant 1 : i8
    %c0_i16 = arith.constant 0 : i16
    %c0_i8 = arith.constant 0 : i8
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c1_i16 = arith.constant 1 : i16
    %c32_i16 = arith.constant 32 : i16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant -1.280000e+02 : f32
    %cst_3 = arith.constant 1.270000e+02 : f32
    %c32 = arith.constant 32 : index
    %0 = npux.alloc : memref<1x32xi8>
    %1 = memref.load %arg1[] : memref<f32>
    %2 = memref.load %arg2[] : memref<i8>
    %3 = arith.extsi %2 : i8 to i32
    %4 = arith.sitofp %3 : i32 to f32
    %5 = npux.alloc : memref<1xindex>
    memref.store %c32, %5[%c0] : memref<1xindex>
    %reshape = memref.reshape %arg0(%5) : (memref<1x32xf32>, memref<1xindex>) -> memref<32xf32>
    npux.free %5 : memref<1xindex>
    %6 = npux.alloc : memref<1xindex>
    memref.store %c32, %6[%c0] : memref<1xindex>
    %reshape_4 = memref.reshape %0(%6) : (memref<1x32xi8>, memref<1xindex>) -> memref<32xi8>
    npux.free %6 : memref<1xindex>
    scf.for %arg5 = %c0 to %c32 step %c1 {
      %13 = memref.load %reshape[%arg5] : memref<32xf32>
      %14 = arith.divf %13, %1 : f32
      %15 = math.floor %14 : f32
      %16 = arith.subf %14, %15 : f32
      %17 = arith.cmpf ogt, %16, %cst : f32
      %18 = arith.addf %15, %cst_1 : f32
      %19 = arith.select %17, %18, %15 : f32
      %20 = arith.mulf %15, %cst : f32
      %21 = math.floor %20 : f32
      %22 = arith.mulf %21, %cst_0 : f32
      %23 = arith.subf %15, %22 : f32
      %24 = arith.cmpf oeq, %23, %cst_1 : f32
      %25 = arith.select %24, %18, %15 : f32
      %26 = arith.cmpf oeq, %16, %cst : f32
      %27 = arith.select %26, %25, %19 : f32
      %28 = arith.addf %27, %4 : f32
      %29 = arith.maxnumf %28, %cst_2 : f32
      %30 = arith.minnumf %29, %cst_3 : f32
      %31 = arith.fptosi %30 : f32 to i32
      %32 = arith.trunci %31 : i32 to i8
      memref.store %32, %reshape_4[%arg5] : memref<32xi8>
    }
    %7 = npux.alloc : memref<1x32xi8>
    %8 = npux.sram_alloc : memref<1x32xi8, 2>
    npux.dma_mvin %0, %8 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.free %0 : memref<1x32xi8>
    npux.sfu_run layernorm %c8_i8, %true in(%8) shape(%c32_i16 x %c1_i16) out(%8) quant(%c0_i32, %c0_i16, %c23890_i16, %c-20_i16, %c22131_i16, %c-9_i16) : memref<1x32xi8, 2>, memref<1x32xi8, 2>
    npux.dma_mvout %7, %8 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.sram_free %8 : memref<1x32xi8, 2>
    %9 = npux.alloc : memref<1x32xi8>
    %10 = npux.sram_alloc : memref<1x32xi8, 2>
    npux.dma_mvin %7, %10 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.free %7 : memref<1x32xi8>
    npux.sfu_run gelu %c8_i8, %true in(%10) shape(%c32_i16 x %c1_i16) out(%10) quant(%c0_i32, %c0_i16, %c24259_i16, %c-20_i16, %c22167_i16, %c-9_i16) : memref<1x32xi8, 2>, memref<1x32xi8, 2>
    npux.dma_mvout %9, %10 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.sram_free %10 : memref<1x32xi8, 2>
    %11 = npux.alloc : memref<1x32xi8>
    %12 = npux.sram_alloc : memref<1x32xi8, 2>
    npux.dma_mvin %9, %12 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.free %9 : memref<1x32xi8>
    npux.sfu_run softmax %c8_i8, %true in(%12) shape(%c32_i16 x %c1_i16) out(%12) quant(%c0_i32, %c0_i16, %c24219_i16, %c-20_i16, %c32512_i16, %c-8_i16) : memref<1x32xi8, 2>, memref<1x32xi8, 2>
    npux.dma_mvout %11, %12 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.sram_free %12 : memref<1x32xi8, 2>
    scf.for %arg5 = %c0 to %c32 step %c1 {
      %13 = memref.load %11[%c0, %arg5] : memref<1x32xi8>
      %14 = memref.load %arg3[] : memref<f32>
      %15 = memref.load %arg2[] : memref<i8>
      %16 = arith.extsi %13 : i8 to i32
      %17 = arith.sitofp %16 : i32 to f32
      %18 = arith.extsi %15 : i8 to i32
      %19 = arith.sitofp %18 : i32 to f32
      %20 = arith.subf %17, %19 : f32
      %21 = arith.mulf %20, %14 : f32
      memref.store %21, %arg4[%c0, %arg5] : memref<1x32xf32>
    }
    npux.free %11 : memref<1x32xi8>
    return
  }
}

