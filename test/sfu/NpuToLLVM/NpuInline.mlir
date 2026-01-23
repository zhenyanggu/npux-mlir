module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "act_test_quant_sym"} {
  func.func @main_graph(%arg0: memref<1x32xf32> {onnx.name = "input"}) -> (memref<1x32xf32> {onnx.name = "output"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.270000e+02 : f32
    %cst_0 = arith.constant -1.280000e+02 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 2.000000e+00 : f32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %c32_i16 = arith.constant 32 : i16
    %c1_i16 = arith.constant 1 : i16
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c0_i16 = arith.constant 0 : i16
    %c1_i8 = arith.constant 1 : i8
    %c23890_i16 = arith.constant 23890 : i16
    %c-20_i16 = arith.constant -20 : i16
    %c22131_i16 = arith.constant 22131 : i16
    %c-9_i16 = arith.constant -9 : i16
    %c8_i8 = arith.constant 8 : i8
    %true = arith.constant true
    %c24259_i16 = arith.constant 24259 : i16
    %c22167_i16 = arith.constant 22167 : i16
    %c24219_i16 = arith.constant 24219 : i16
    %c32512_i16 = arith.constant 32512 : i16
    %c-8_i16 = arith.constant -8 : i16
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
      %19 = memref.load %arg0[%c0, %arg1] : memref<1x32xf32>
      %20 = memref.load %1[%arg1] : memref<32xf32>
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc[%c0, %arg1] : memref<1x32xf32>
    }
    %alloc_4 = memref.alloc() : memref<1x32xf32>
    %6 = npux.alloc : memref<1x32xi8>
    %7 = memref.load %4[] : memref<f32>
    %8 = memref.load %3[] : memref<i8>
    %9 = arith.extsi %8 : i8 to i32
    %10 = arith.sitofp %9 : i32 to f32
    %11 = npux.alloc : memref<1xindex>
    memref.store %c32, %11[%c0] : memref<1xindex>
    %reshape = memref.reshape %alloc(%11) : (memref<1x32xf32>, memref<1xindex>) -> memref<32xf32>
    npux.free %11 : memref<1xindex>
    %12 = npux.alloc : memref<1xindex>
    memref.store %c32, %12[%c0] : memref<1xindex>
    %reshape_5 = memref.reshape %6(%12) : (memref<1x32xi8>, memref<1xindex>) -> memref<32xi8>
    npux.free %12 : memref<1xindex>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %19 = memref.load %reshape[%arg1] : memref<32xf32>
      %20 = arith.divf %19, %7 : f32
      %21 = math.floor %20 : f32
      %22 = arith.subf %20, %21 : f32
      %23 = arith.cmpf ogt, %22, %cst_3 : f32
      %24 = arith.addf %21, %cst_1 : f32
      %25 = arith.select %23, %24, %21 : f32
      %26 = arith.mulf %21, %cst_3 : f32
      %27 = math.floor %26 : f32
      %28 = arith.mulf %27, %cst_2 : f32
      %29 = arith.subf %21, %28 : f32
      %30 = arith.cmpf oeq, %29, %cst_1 : f32
      %31 = arith.select %30, %24, %21 : f32
      %32 = arith.cmpf oeq, %22, %cst_3 : f32
      %33 = arith.select %32, %31, %25 : f32
      %34 = arith.addf %33, %10 : f32
      %35 = arith.maxnumf %34, %cst_0 : f32
      %36 = arith.minnumf %35, %cst : f32
      %37 = arith.fptosi %36 : f32 to i32
      %38 = arith.trunci %37 : i32 to i8
      memref.store %38, %reshape_5[%arg1] : memref<32xi8>
    }
    %13 = npux.alloc : memref<1x32xi8>
    %14 = npux.sram_alloc {npu.offset = 0 : i32} : memref<1x32xi8, 2>
    npux.dma_mvin %6, %14 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.free %6 : memref<1x32xi8>
    npux.sfu_run layernorm %c8_i8, %true in(%14) shape(%c32_i16 x %c1_i16) out(%14) quant(%c0_i32, %c0_i16, %c23890_i16, %c-20_i16, %c22131_i16, %c-9_i16) : memref<1x32xi8, 2>, memref<1x32xi8, 2>
    npux.dma_mvout %13, %14 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.sram_free %14 : memref<1x32xi8, 2>
    %15 = npux.alloc : memref<1x32xi8>
    %16 = npux.sram_alloc {npu.offset = 0 : i32} : memref<1x32xi8, 2>
    npux.dma_mvin %13, %16 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.free %13 : memref<1x32xi8>
    npux.sfu_run gelu %c8_i8, %true in(%16) shape(%c32_i16 x %c1_i16) out(%16) quant(%c0_i32, %c0_i16, %c24259_i16, %c-20_i16, %c22167_i16, %c-9_i16) : memref<1x32xi8, 2>, memref<1x32xi8, 2>
    npux.dma_mvout %15, %16 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.sram_free %16 : memref<1x32xi8, 2>
    %17 = npux.alloc : memref<1x32xi8>
    %18 = npux.sram_alloc {npu.offset = 0 : i32} : memref<1x32xi8, 2>
    npux.dma_mvin %15, %18 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.free %15 : memref<1x32xi8>
    npux.sfu_run softmax %c8_i8, %true in(%18) shape(%c32_i16 x %c1_i16) out(%18) quant(%c0_i32, %c0_i16, %c24219_i16, %c-20_i16, %c32512_i16, %c-8_i16) : memref<1x32xi8, 2>, memref<1x32xi8, 2>
    npux.dma_mvout %17, %18 shape(%c32_i16 x %c1_i16) stride(%c32_i16, %c32_i16) cfg(%c1_i8, %c0_i8, %c0_i8) quant(%false, %c0_i32, %c0_i16, %c0_i16) : memref<1x32xi8>, memref<1x32xi8, 2>
    npux.sram_free %18 : memref<1x32xi8, 2>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %19 = memref.load %17[%c0, %arg1] : memref<1x32xi8>
      %20 = memref.load %5[] : memref<f32>
      %21 = memref.load %3[] : memref<i8>
      %22 = arith.extsi %19 : i8 to i32
      %23 = arith.sitofp %22 : i32 to f32
      %24 = arith.extsi %21 : i8 to i32
      %25 = arith.sitofp %24 : i32 to f32
      %26 = arith.subf %23, %25 : f32
      %27 = arith.mulf %26, %20 : f32
      memref.store %27, %alloc_4[%c0, %arg1] : memref<1x32xf32>
    }
    npux.free %17 : memref<1x32xi8>
    memref.dealloc %alloc : memref<1x32xf32>
    %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<1x32xf32>
    scf.for %arg1 = %c0 to %c32 step %c1 {
      %19 = memref.load %alloc_4[%c0, %arg1] : memref<1x32xf32>
      %20 = memref.load %2[%arg1] : memref<32xf32>
      %21 = arith.addf %19, %20 : f32
      memref.store %21, %alloc_6[%c0, %arg1] : memref<1x32xf32>
    }
    memref.dealloc %alloc_4 : memref<1x32xf32>
    "npux.destroy"() : () -> ()
    return %alloc_6 : memref<1x32xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22output\22 }\0A\0A]\00"} : () -> ()
}

