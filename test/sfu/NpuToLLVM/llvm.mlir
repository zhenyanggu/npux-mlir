module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "act_test_quant_sym"} {
  llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.mlir.global external constant @_entry_point_1_act_test_quant_sym("run_main_graph_act_test_quant_sym\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_in_sig_act_test_quant_sym("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22input\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_out_sig_act_test_quant_sym("[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22output\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_act_test_quant_sym("run_main_graph\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_in_sig_act_test_quant_sym("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22input\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_out_sig_act_test_quant_sym("[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 32] , \22name\22 : \22output\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.func @omGetExternalConstantAddr(!llvm.ptr, !llvm.ptr, i64)
  llvm.func @omMMapBinaryFile(!llvm.ptr, !llvm.ptr, i64, i64) -> i1
  llvm.func @omTensorListGetSize(!llvm.ptr) -> i64
  llvm.func @omTensorPrint(!llvm.ptr, !llvm.ptr)
  llvm.func @omTensorListGetOmtArray(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorSetDataType(!llvm.ptr, i64)
  llvm.func @omTensorGetDataType(!llvm.ptr) -> i64
  llvm.func @omTensorGetStrides(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetShape(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetRank(!llvm.ptr) -> i64
  llvm.func @omTensorSetDataPtr(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr)
  llvm.func @omTensorGetDataPtr(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorDestroy(!llvm.ptr)
  llvm.func @omTensorCreateUntyped(i64) -> !llvm.ptr
  llvm.func @omTensorListCreate(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @npu_destroy()
  llvm.func @free(!llvm.ptr)
  llvm.func @npu_dma_mvout(!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)
  llvm.func @npu_sfu_run(i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16)
  llvm.func @npu_dma_mvin(!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16)
  llvm.func @npu_mem_free(!llvm.ptr)
  llvm.func @npu_mem_alloc(i64) -> !llvm.ptr
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @constant_4_act_test_quant_sym(dense<0.00787401571> : tensor<f32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<1 x f32>
  llvm.mlir.global internal constant @constant_3_act_test_quant_sym(dense<0.0227836408> : tensor<f32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<1 x f32>
  llvm.mlir.global internal constant @constant_2_act_test_quant_sym(dense<0> : tensor<i8>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<1 x i8>
  llvm.mlir.global internal constant @constant_1_act_test_quant_sym(dense<[-0.174824119, 0.0794920623, -0.0248840358, -0.109353125, -0.0772407278, -0.0361809246, 0.219115898, 0.187830746, 0.245069295, -0.0233103242, 0.0372669809, -0.0510918386, 0.044748459, -0.00205091923, -0.0826271325, -0.157617912, 0.100945644, 0.106719315, 0.0168068167, -0.0849939808, -0.0280704256, -0.0105170989, 0.18064329, -0.0139956269, -0.035048712, 0.0254881084, -0.0581126623, -0.12114203, 0.0639609322, -0.0395348035, -0.140444294, -0.0503093414]> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<32 x f32>
  llvm.mlir.global internal constant @constant_0_act_test_quant_sym(dense<[-9.82335652E-4, -0.0923864468, 0.0308713373, 0.125592053, -0.0481653772, 0.0492569357, -0.0528857373, 0.191135094, -0.102464393, 0.0097812349, -0.0817779526, 0.00841632765, -0.0643123686, -0.125691578, -0.048106797, -0.115775935, -0.125293836, 7.638620e-02, 0.0334553234, -0.0247902982, 0.0771105587, 0.103332363, -0.129471868, 0.00529868389, 0.0113818403, -0.0332146063, 0.0631280169, 0.0246195775, -0.123717859, -0.0606194735, 0.0602609515, -0.0785476937]> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<32 x f32>
  llvm.func @npu_init() -> i32
  llvm.func @main_graph_act_test_quant_sym(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(2 : i8) : i8
    %1 = llvm.mlir.constant(8 : i64) : i64
    %2 = llvm.mlir.constant(32 : i64) : i64
    %3 = llvm.mlir.constant(16 : index) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.addressof @constant_4_act_test_quant_sym : !llvm.ptr
    %6 = llvm.mlir.addressof @constant_3_act_test_quant_sym : !llvm.ptr
    %7 = llvm.mlir.addressof @constant_2_act_test_quant_sym : !llvm.ptr
    %8 = llvm.mlir.addressof @constant_1_act_test_quant_sym : !llvm.ptr
    %9 = llvm.mlir.addressof @constant_0_act_test_quant_sym : !llvm.ptr
    %10 = llvm.mlir.constant(32 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(-8 : i16) : i16
    %14 = llvm.mlir.constant(32512 : i16) : i16
    %15 = llvm.mlir.constant(24219 : i16) : i16
    %16 = llvm.mlir.constant(22167 : i16) : i16
    %17 = llvm.mlir.constant(24259 : i16) : i16
    %18 = llvm.mlir.constant(true) : i1
    %19 = llvm.mlir.constant(8 : i8) : i8
    %20 = llvm.mlir.constant(-9 : i16) : i16
    %21 = llvm.mlir.constant(22131 : i16) : i16
    %22 = llvm.mlir.constant(-20 : i16) : i16
    %23 = llvm.mlir.constant(23890 : i16) : i16
    %24 = llvm.mlir.constant(1 : i8) : i8
    %25 = llvm.mlir.constant(0 : i16) : i16
    %26 = llvm.mlir.constant(0 : i8) : i8
    %27 = llvm.mlir.constant(false) : i1
    %28 = llvm.mlir.constant(0 : i32) : i32
    %29 = llvm.mlir.constant(1 : i16) : i16
    %30 = llvm.mlir.constant(32 : i16) : i16
    %31 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %32 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %33 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %34 = llvm.mlir.constant(-1.280000e+02 : f32) : f32
    %35 = llvm.mlir.constant(1.270000e+02 : f32) : f32
    %36 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.call @npu_init() : () -> i32
    %38 = llvm.getelementptr %4[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.add %39, %3 : i64
    %41 = llvm.call @malloc(%40) : (i64) -> !llvm.ptr
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.sub %3, %11 : i64
    %44 = llvm.add %42, %43 : i64
    %45 = llvm.urem %44, %3 : i64
    %46 = llvm.sub %44, %45 : i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    llvm.br ^bb1(%12 : i64)
  ^bb1(%48: i64):  // 2 preds: ^bb0, ^bb2
    %49 = llvm.icmp "slt" %48, %10 : i64
    llvm.cond_br %49, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %50 = llvm.mul %12, %10 overflow<nsw, nuw> : i64
    %51 = llvm.add %50, %48 overflow<nsw, nuw> : i64
    %52 = llvm.getelementptr inbounds|nuw %arg1[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.load %52 : !llvm.ptr -> f32
    %54 = llvm.getelementptr inbounds|nuw %9[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %55 = llvm.load %54 : !llvm.ptr -> f32
    %56 = llvm.fadd %53, %55 : f32
    %57 = llvm.mul %12, %10 overflow<nsw, nuw> : i64
    %58 = llvm.add %57, %48 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds|nuw %47[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %59 : f32, !llvm.ptr
    %60 = llvm.add %48, %11 : i64
    llvm.br ^bb1(%60 : i64)
  ^bb3:  // pred: ^bb1
    %61 = llvm.getelementptr %4[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.call @malloc(%62) : (i64) -> !llvm.ptr
    %64 = llvm.call @npu_mem_alloc(%2) : (i64) -> !llvm.ptr
    %65 = llvm.load %6 : !llvm.ptr -> f32
    %66 = llvm.load %7 : !llvm.ptr -> i8
    %67 = llvm.sext %66 : i8 to i32
    %68 = llvm.sitofp %67 : i32 to f32
    %69 = llvm.call @npu_mem_alloc(%1) : (i64) -> !llvm.ptr
    llvm.store %10, %69 : i64, !llvm.ptr
    llvm.call @npu_mem_free(%69) : (!llvm.ptr) -> ()
    %70 = llvm.call @npu_mem_alloc(%1) : (i64) -> !llvm.ptr
    llvm.store %10, %70 : i64, !llvm.ptr
    llvm.call @npu_mem_free(%70) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%12 : i64)
  ^bb4(%71: i64):  // 2 preds: ^bb3, ^bb5
    %72 = llvm.icmp "slt" %71, %10 : i64
    llvm.cond_br %72, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %73 = llvm.getelementptr inbounds|nuw %47[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %74 = llvm.load %73 : !llvm.ptr -> f32
    %75 = llvm.fdiv %74, %65 : f32
    %76 = llvm.intr.floor(%75) : (f32) -> f32
    %77 = llvm.fsub %75, %76 : f32
    %78 = llvm.fcmp "ogt" %77, %31 : f32
    %79 = llvm.fadd %76, %33 : f32
    %80 = llvm.select %78, %79, %76 : i1, f32
    %81 = llvm.fmul %76, %31 : f32
    %82 = llvm.intr.floor(%81) : (f32) -> f32
    %83 = llvm.fmul %82, %32 : f32
    %84 = llvm.fsub %76, %83 : f32
    %85 = llvm.fcmp "oeq" %84, %33 : f32
    %86 = llvm.select %85, %79, %76 : i1, f32
    %87 = llvm.fcmp "oeq" %77, %31 : f32
    %88 = llvm.select %87, %86, %80 : i1, f32
    %89 = llvm.fadd %88, %68 : f32
    %90 = llvm.intr.maxnum(%89, %34) : (f32, f32) -> f32
    %91 = llvm.intr.minnum(%90, %35) : (f32, f32) -> f32
    %92 = llvm.fptosi %91 : f32 to i32
    %93 = llvm.trunc %92 : i32 to i8
    %94 = llvm.getelementptr inbounds|nuw %64[%71] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %93, %94 : i8, !llvm.ptr
    %95 = llvm.add %71, %11 : i64
    llvm.br ^bb4(%95 : i64)
  ^bb6:  // pred: ^bb4
    %96 = llvm.call @npu_mem_alloc(%2) : (i64) -> !llvm.ptr
    llvm.call @npu_dma_mvin(%64, %28, %30, %29, %30, %30, %24, %26, %26, %27, %28, %25, %25) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    llvm.call @npu_mem_free(%64) : (!llvm.ptr) -> ()
    llvm.call @npu_sfu_run(%0, %19, %18, %28, %30, %29, %28, %28, %25, %23, %22, %21, %20) : (i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) -> ()
    llvm.call @npu_dma_mvout(%96, %28, %30, %29, %30, %30, %24, %26, %26, %27, %28, %25, %25) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    %97 = llvm.call @npu_mem_alloc(%2) : (i64) -> !llvm.ptr
    llvm.call @npu_dma_mvin(%96, %28, %30, %29, %30, %30, %24, %26, %26, %27, %28, %25, %25) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    llvm.call @npu_mem_free(%96) : (!llvm.ptr) -> ()
    llvm.call @npu_sfu_run(%24, %19, %18, %28, %30, %29, %28, %28, %25, %17, %22, %16, %20) : (i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) -> ()
    llvm.call @npu_dma_mvout(%97, %28, %30, %29, %30, %30, %24, %26, %26, %27, %28, %25, %25) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    %98 = llvm.call @npu_mem_alloc(%2) : (i64) -> !llvm.ptr
    llvm.call @npu_dma_mvin(%97, %28, %30, %29, %30, %30, %24, %26, %26, %27, %28, %25, %25) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    llvm.call @npu_mem_free(%97) : (!llvm.ptr) -> ()
    llvm.call @npu_sfu_run(%26, %19, %18, %28, %30, %29, %28, %28, %25, %15, %22, %14, %13) : (i8, i8, i1, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16) -> ()
    llvm.call @npu_dma_mvout(%98, %28, %30, %29, %30, %30, %24, %26, %26, %27, %28, %25, %25) : (!llvm.ptr, i32, i16, i16, i16, i16, i8, i8, i8, i1, i32, i16, i16) -> ()
    llvm.br ^bb7(%12 : i64)
  ^bb7(%99: i64):  // 2 preds: ^bb6, ^bb8
    %100 = llvm.icmp "slt" %99, %10 : i64
    llvm.cond_br %100, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %101 = llvm.mul %12, %10 overflow<nsw, nuw> : i64
    %102 = llvm.add %101, %99 overflow<nsw, nuw> : i64
    %103 = llvm.getelementptr inbounds|nuw %98[%102] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %104 = llvm.load %103 : !llvm.ptr -> i8
    %105 = llvm.load %5 : !llvm.ptr -> f32
    %106 = llvm.load %7 : !llvm.ptr -> i8
    %107 = llvm.sext %104 : i8 to i32
    %108 = llvm.sitofp %107 : i32 to f32
    %109 = llvm.sext %106 : i8 to i32
    %110 = llvm.sitofp %109 : i32 to f32
    %111 = llvm.fsub %108, %110 : f32
    %112 = llvm.fmul %111, %105 : f32
    %113 = llvm.mul %12, %10 overflow<nsw, nuw> : i64
    %114 = llvm.add %113, %99 overflow<nsw, nuw> : i64
    %115 = llvm.getelementptr inbounds|nuw %63[%114] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %112, %115 : f32, !llvm.ptr
    %116 = llvm.add %99, %11 : i64
    llvm.br ^bb7(%116 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @npu_mem_free(%98) : (!llvm.ptr) -> ()
    llvm.call @free(%41) : (!llvm.ptr) -> ()
    %117 = llvm.getelementptr %4[32] : (!llvm.ptr) -> !llvm.ptr, f32
    %118 = llvm.ptrtoint %117 : !llvm.ptr to i64
    %119 = llvm.add %118, %3 : i64
    %120 = llvm.call @malloc(%119) : (i64) -> !llvm.ptr
    %121 = llvm.ptrtoint %120 : !llvm.ptr to i64
    %122 = llvm.sub %3, %11 : i64
    %123 = llvm.add %121, %122 : i64
    %124 = llvm.urem %123, %3 : i64
    %125 = llvm.sub %123, %124 : i64
    %126 = llvm.inttoptr %125 : i64 to !llvm.ptr
    %127 = llvm.insertvalue %120, %36[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.insertvalue %126, %127[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %12, %128[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %11, %129[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %10, %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %10, %131[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %11, %132[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb10(%12 : i64)
  ^bb10(%134: i64):  // 2 preds: ^bb9, ^bb11
    %135 = llvm.icmp "slt" %134, %10 : i64
    llvm.cond_br %135, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %136 = llvm.mul %12, %10 overflow<nsw, nuw> : i64
    %137 = llvm.add %136, %134 overflow<nsw, nuw> : i64
    %138 = llvm.getelementptr inbounds|nuw %63[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %139 = llvm.load %138 : !llvm.ptr -> f32
    %140 = llvm.getelementptr inbounds|nuw %8[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %141 = llvm.load %140 : !llvm.ptr -> f32
    %142 = llvm.fadd %139, %141 : f32
    %143 = llvm.mul %12, %10 overflow<nsw, nuw> : i64
    %144 = llvm.add %143, %134 overflow<nsw, nuw> : i64
    %145 = llvm.getelementptr inbounds|nuw %126[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %142, %145 : f32, !llvm.ptr
    %146 = llvm.add %134, %11 : i64
    llvm.br ^bb10(%146 : i64)
  ^bb12:  // pred: ^bb10
    llvm.call @free(%63) : (!llvm.ptr) -> ()
    llvm.call @npu_destroy() : () -> ()
    llvm.return %133 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_main_graph_act_test_quant_sym(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.call @main_graph_act_test_quant_sym(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %8, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @run_main_graph_act_test_quant_sym(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(2 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %5 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    %6 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
    %7 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    %8 = llvm.call @omTensorGetDataPtr(%6) : (!llvm.ptr) -> !llvm.ptr
    %9 = llvm.insertvalue %8, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %1, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.call @omTensorGetShape(%6) : (!llvm.ptr) -> !llvm.ptr
    %13 = llvm.call @omTensorGetStrides(%6) : (!llvm.ptr) -> !llvm.ptr
    %14 = llvm.load %12 : !llvm.ptr -> i64
    %15 = llvm.insertvalue %14, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %13 : !llvm.ptr -> i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %19 = llvm.load %18 : !llvm.ptr -> i64
    %20 = llvm.insertvalue %19, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.getelementptr %13[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.insertvalue %22, %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %23, %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_main_graph_act_test_quant_sym(%5, %7) : (!llvm.ptr, !llvm.ptr) -> ()
    %24 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.alloca %3 x !llvm.ptr : (i64) -> !llvm.ptr
    %26 = llvm.call @omTensorCreateUntyped(%0) : (i64) -> !llvm.ptr
    %27 = llvm.extractvalue %24[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @omTensorSetDataPtr(%26, %3, %27, %28) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%26, %3) : (!llvm.ptr, i64) -> ()
    %29 = llvm.call @omTensorGetShape(%26) : (!llvm.ptr) -> !llvm.ptr
    %30 = llvm.call @omTensorGetStrides(%26) : (!llvm.ptr) -> !llvm.ptr
    %31 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %31, %29 : i64, !llvm.ptr
    %32 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %32, %30 : i64, !llvm.ptr
    %33 = llvm.extractvalue %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.getelementptr %29[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %33, %34 : i64, !llvm.ptr
    %35 = llvm.extractvalue %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.getelementptr %30[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %35, %36 : i64, !llvm.ptr
    llvm.store %26, %25 : !llvm.ptr, !llvm.ptr
    %37 = llvm.call @omTensorListCreate(%25, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.return %37 : !llvm.ptr
  }
  llvm.func @run_main_graph(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @run_main_graph_act_test_quant_sym(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global internal constant @_entry_point_arrays_act_test_quant_sym() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_act_test_quant_sym : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @_entry_point_0_act_test_quant_sym : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.func @omQueryEntryPoints_act_test_quant_sym(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @_entry_point_arrays_act_test_quant_sym : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "ne" %arg0, %2 : !llvm.ptr
    llvm.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.store %1, %arg0 : i64, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omQueryEntryPoints(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omQueryEntryPoints_act_test_quant_sym(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature_act_test_quant_sym(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_in_sig_act_test_quant_sym : !llvm.ptr
    %2 = llvm.mlir.constant(34 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_act_test_quant_sym : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_in_sig_act_test_quant_sym : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_act_test_quant_sym : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %4 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %10 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %11 = llvm.icmp "eq" %10, %6 : i32
    llvm.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.return %1 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omInputSignature_act_test_quant_sym(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature_act_test_quant_sym(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_out_sig_act_test_quant_sym : !llvm.ptr
    %2 = llvm.mlir.constant(34 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_act_test_quant_sym : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_out_sig_act_test_quant_sym : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_act_test_quant_sym : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %4 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %10 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %11 = llvm.icmp "eq" %10, %6 : i32
    llvm.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.return %1 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omOutputSignature_act_test_quant_sym(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}

